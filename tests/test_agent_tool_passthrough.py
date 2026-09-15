"""端到端验证：Agent -> 工具 -> RunnableConfig -> JWT。

用一个「脚本化的假大模型」替代真实 LLM：第一次返回一个工具调用请求，
第二次返回最终文本。这样可以完整走通
``create_agent`` 的真实工具执行链路，验证：

1. 传给 ``agent.ainvoke(config=...)`` 的 ``jwt_token`` 能否到达工具函数；
2. 工具签名中的 ``runtime`` / ``config`` 参数是否**不会**出现在模型可见的
   tool schema 中（模型无法看到、更无法伪造 JWT）。

运行::

    python tests/test_agent_tool_passthrough.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.prebuilt import ToolRuntime  # noqa: E402

CAPTURED: dict[str, Any] = {}


@tool("probe_jwt")
def probe_jwt(marker: str, runtime: ToolRuntime) -> str:
    """测试用工具：读取 RunnableConfig 中的 jwt_token。

    Args:
        marker: 任意标记文本。
        runtime: 由 LangGraph 注入的运行时对象，模型不可见。
    """
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") or {}
    CAPTURED["jwt"] = configurable.get("jwt_token")
    CAPTURED["session_id"] = configurable.get("session_id")
    CAPTURED["marker"] = marker
    CAPTURED["tool_call_id"] = getattr(runtime, "tool_call_id", None)
    return f"probe ok: marker={marker}"


class ScriptedChatModel(BaseChatModel):
    """按脚本返回固定响应的假模型：先请求工具，再给最终回答。"""

    tool_name: str = "probe_jwt"
    tool_args: dict[str, Any] = {"marker": "hello"}

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        # 保持原样返回自身：我们不需要真正的 tool schema 绑定
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        has_tool_result = any(isinstance(message, ToolMessage) for message in messages)

        if not has_tool_result:
            # 第一轮：请求调用工具
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_name,
                        "args": dict(self.tool_args),
                        "id": "call_probe_1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            # 第二轮：基于工具结果给出最终回答
            tool_output = ""
            for message_item in reversed(messages):
                if isinstance(message_item, ToolMessage):
                    tool_output = str(message_item.content)
                    break
            message = AIMessage(content=f"工具已执行：{tool_output}")

        return ChatResult(generations=[ChatGeneration(message=message)])


def main() -> int:
    from langchain.agents import create_agent

    agent = create_agent(
        model=ScriptedChatModel(),
        tools=[probe_jwt],
        system_prompt="你是测试助手。",
    )

    # --- 1. 检查模型可见的 schema 不含 runtime/config ---
    print("--- 模型可见的 tool schema ---")
    for item in agent.get_input_jsonschema().get("properties", {}):
        pass
    from langchain_core.utils.function_calling import convert_to_openai_tool

    oai_schema = convert_to_openai_tool(probe_jwt)
    params = oai_schema.get("function", {}).get("parameters", {})
    properties = params.get("properties", {})
    print("exposed args:", list(properties))
    schema_ok = "runtime" not in properties and "config" not in properties

    # --- 2. 真实走一遍 agent -> tool ---
    config = {
        "configurable": {
            "jwt_token": "eyJhbGciOiJIUzI1NiJ9.PAYLOAD.SIGNATURE",
            "session_id": "e2e-session",
            "thread_id": "e2e-thread",
        },
        "recursion_limit": 10,
    }

    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "请调用工具"}]}, config=config)
    )

    final = result["messages"][-1]
    print("\n--- Agent 最终回答 ---")
    print(final.content)

    print("\n--- 工具捕获到的运行时数据 ---")
    for key, value in CAPTURED.items():
        print(f"  {key} = {value}")

    passed = True
    if not schema_ok:
        print("FAIL: runtime/config 暴露给了模型")
        passed = False
    else:
        print("PASS: 模型的 tool schema 只包含业务参数")

    if CAPTURED.get("jwt") == "eyJhbGciOiJIUzI1NiJ9.PAYLOAD.SIGNATURE":
        print("PASS: JWT 经 RunnableConfig 成功透传到工具")
    else:
        print(f"FAIL: JWT 未透传（实际 {CAPTURED.get('jwt')!r}）")
        passed = False

    if CAPTURED.get("session_id") == "e2e-session":
        print("PASS: session_id 一并透传")
    else:
        print(f"FAIL: session_id 未透传（实际 {CAPTURED.get('session_id')!r}）")
        passed = False

    if CAPTURED.get("tool_call_id"):
        print("PASS: tool_call_id 已注入")
    else:
        print("WARN: tool_call_id 为空")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
