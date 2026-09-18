"""LangChain Agent：自主决策调用政策检索工具或投诉工单工具。

决策逻辑完全交给模型（ReAct 风格的工具调用循环），提示词只负责界定边界：

* 问「规定是什么」-> 调 ``search_platform_policy``
* 说「我要投诉某件具体的事」-> 调 ``create_complaint_ticket``
* 问「我的投诉进度」-> 调 ``list_my_complaints``
* 闲聊/问候 -> 直接回答，不调工具

用户 JWT 通过 ``RunnableConfig`` 贯穿到工具，模型看不到也改不了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger
from app.core.request_context import get_jwt_from_config
from app.tools.complaint_tools import get_complaint_tools
from app.tools.policy_tools import get_policy_tools

logger = get_logger(__name__)

AGENT_SYSTEM_PROMPT = """你是「拾汇商城」的官方智能客服助手「小汇」，通过工具为用户提供服务。

你有以下能力，请根据用户意图**自主判断**该用哪个：

1. `search_platform_policy` —— 检索平台政策知识库。
   适用：询问平台**规则、政策、流程、时限、费用标准**类问题。
   例如：退货政策、退款要多久、运费怎么算、能否开发票、会员权益、优惠券规则、
   配送范围与时效、价保规则、售后条件等。

2. `create_complaint_ticket` —— 创建投诉工单，转人工跟进。
   适用：用户要**投诉/举报/追责**某件**具体已发生的事**（买到假货、收到破损商品、
   商家态度恶劣、迟迟不发货、拒绝退款等），需要平台介入处理。
   注意：咨询政策不等于投诉。用户只是问「退货政策是什么」时用工具 1，不要建单。
   用户只说「我要投诉」但没说原因时，先用一句话追问具体情况，**不要**直接建单。

3. `list_my_complaints` —— 查询当前用户自己的投诉工单与处理进度。

【回答规则】
- 涉及平台政策的回答，**必须**先调用 `search_platform_policy` 获取资料，
  且只能依据检索到的片段作答。资料中没有的内容一律不回答，要明确告知用户
  「知识库中没有相关资料」，并建议联系人工客服。严禁编造条款、时限、金额、
  条款编号或生效日期。
- 依据资料作答时，用 [编号] 标注来源。
- 创建工单后，如实告知工单号与后续处理方式；**不要**承诺具体的赔偿方案或处理结果。
- 工单创建失败时，如实说明失败原因，**绝对不要**谎称已经创建成功。
- 用简体中文，语气礼貌、专业、简洁。先给结论，再给要点。
- 不要向用户暴露任何技术实现细节（工具名、检索、向量、JWT、接口等）。
- 不要提及、询问或尝试获取任何用户身份标识；工单归属系统会自动处理。
"""

# 工具名 -> 面向用户的动作描述（用于前端展示「正在做什么」）
TOOL_ACTION_LABELS: dict[str, str] = {
    "search_platform_policy": "正在检索平台政策知识库",
    "create_complaint_ticket": "正在创建投诉工单",
    "list_my_complaints": "正在查询您的投诉工单",
}


@dataclass
class AgentStep:
    """一次工具调用记录，用于可观测性与前端展示。"""

    tool: str
    label: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "label": self.label, "arguments": self.arguments}


@dataclass
class AgentResult:
    """Agent 一轮对话的返回结构。"""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)
    session_id: str = ""
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [step.to_dict() for step in self.steps],
            "used_tools": self.used_tools,
            "session_id": self.session_id,
            "message_count": self.message_count,
        }


def _extract_text(message: BaseMessage) -> str:
    """从消息对象提取纯文本（兼容 content 为字符串或内容块列表）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


class CustomerServiceAgent:
    """智能客服 Agent 门面。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._agent: Any = None

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def build(self, streaming : bool = True) -> Any:
        """构建（或复用）Agent。"""
        if self._agent is not None:
            return self._agent

        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        from app.agent.llm import build_chat_model

        tools = [*get_policy_tools(), *get_complaint_tools()]
        model = build_chat_model(self.settings, streaming=streaming)

        # InMemorySaver 提供多轮对话记忆（按 thread_id 隔离）。
        # 单进程部署足够；多副本部署时应换成 Redis/Postgres checkpointer。
        checkpointer = InMemorySaver()

        logger.info("构建 Agent，可用工具: %s", [t.name for t in tools])
        self._agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=checkpointer,
            name="tenhub_customer_service",
        )
        return self._agent

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    def _build_call_config(self, config: RunnableConfig) -> RunnableConfig:
        """补齐递归上限等运行时参数，保留调用方注入的 JWT。"""
        call_config: dict[str, Any] = dict(config)
        call_config.setdefault("recursion_limit", self.settings.agent_recursion_limit)
        return call_config

    @staticmethod
    def _parse_steps(messages: Sequence[BaseMessage]) -> tuple[str, list[AgentStep]]:
        """从消息序列中提取最终回答与工具调用步骤。"""
        steps: list[AgentStep] = []
        final_text = ""

        for message in messages:
            tool_calls = getattr(message, "tool_calls", None) or []
            for call in tool_calls:
                name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
                args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
                if not name:
                    continue
                steps.append(
                    AgentStep(
                        tool=name,
                        label=TOOL_ACTION_LABELS.get(name, f"正在调用 {name}"),
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

            if isinstance(message, AIMessage):
                text = _extract_text(message)
                if text:
                    final_text = text

        return final_text, steps

    async def ainvoke(
        self,
        message: str,
        *,
        config: RunnableConfig,
        session_id: str,
    ) -> AgentResult:
        """异步执行一轮对话。"""
        # 提前校验 JWT 是否已注入，避免工具执行到一半才发现缺失
        get_jwt_from_config(config)

        agent = self.build()
        call_config = self._build_call_config(
            {**config, "configurable": {**config.get("configurable", {}), "thread_id": session_id}}
        )

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=call_config,
        )

        messages: list[BaseMessage] = list(result.get("messages", []))
        answer, steps = self._parse_steps(messages)

        if not answer:
            logger.warning("Agent 未产出文本回答，消息数=%d", len(messages))
            answer = (
                "抱歉，我暂时没能理解您的问题。"
                "您可以换个说法，或直接描述您想咨询的政策或需要投诉的问题。"
            )

        used_tools = list(dict.fromkeys(step.tool for step in steps))
        logger.info(
            "Agent 完成 session=%s tools=%s answer_len=%d",
            session_id,
            used_tools or "无",
            len(answer),
        )

        return AgentResult(
            answer=answer,
            steps=steps,
            used_tools=used_tools,
            session_id=session_id,
            message_count=len(messages),
        )

    async def astream_events(
        self,
        message: str,
        *,
        config: RunnableConfig,
        session_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式执行，产出统一事件供 SSE 推送。

        事件类型：
        * ``tool_start`` —— 开始调用工具（前端可展示「正在检索…」）
        * ``token``     —— 增量文本
        * ``done``      —— 结束，携带完整回答
        """
        get_jwt_from_config(config)
        agent = self.build(True)
        call_config = self._build_call_config(
            {**config, "configurable": {**config.get("configurable", {}), "thread_id": session_id}}
        )

        collected: list[str] = []
        steps: list[AgentStep] = []

        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=call_config,
            version="v2",
        ):
            kind = event.get("event")
            # logger.info(kind)

            if kind == "on_tool_start":
                name = event.get("name", "")
                step = AgentStep(
                    tool=name,
                    label=TOOL_ACTION_LABELS.get(name, f"正在调用 {name}"),
                    arguments=event.get("data", {}).get("input", {}) or {},
                )
                steps.append(step)
                yield {"type": "tool_start", **step.to_dict()}

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                text = _extract_text(chunk) if chunk is not None else ""
                if text:
                    collected.append(text)
                    yield {"type": "token", "content": text}

        answer = "".join(collected).strip()
        if not answer:
            answer = "抱歉，我暂时没能生成有效回复，请换个说法再试一次。"

        yield {
            "type": "done",
            "answer": answer,
            "steps": [step.to_dict() for step in steps],
            "used_tools": list(dict.fromkeys(step.tool for step in steps)),
            "session_id": session_id,
        }


@lru_cache(maxsize=1)
def get_agent() -> CustomerServiceAgent:
    """全局唯一 Agent 实例（保留多轮记忆）。"""
    return CustomerServiceAgent()


# 会话 ID 允许的字符集，避免脏值污染 thread_id
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-:.]{1,128}$")


def normalize_session_id(raw: str | None, *, fallback: str) -> str:
    """校验并规范化会话 ID。"""
    if raw and _SESSION_ID_PATTERN.match(raw):
        return raw
    return fallback
