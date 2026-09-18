"""RAG 政策问答链。

设计目标：**回答必须严格基于检索到的上下文**，检索不到依据时明确拒答，
不允许模型用自身参数化知识编造平台政策。

三道防线
--------
1. **前置门控**：检索结果的向量相似度全部低于 ``relevance_threshold`` 时，
   直接返回固定拒答文案，不调用大模型——既省成本，也杜绝幻觉。
2. **Prompt 约束**：系统提示明确要求「仅依据资料作答」「资料不足必须拒答」
   「不得编造条款编号、时效、金额」。
3. **后置校验**：对模型输出做拒答短语识别，统一 ``answerable`` 标记与
   引用来源，便于上游与前端按统一契约处理。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger
from app.rag.hybrid_retriever import HybridPolicyRetriever, format_documents

logger = get_logger(__name__)

# 检索完全无依据时的标准拒答文案
REFUSAL_NO_CONTEXT = (
    "抱歉，我在平台政策知识库中没有找到与该问题相关的资料，"
    "因此无法给出准确回答。为避免误导，请您补充更具体的信息，"
    "或联系人工客服（客服热线 400-000-0000）进一步咨询。"
)

# 模型主动表示无法回答时的识别模式
_REFUSAL_PATTERNS = (
    r"无法(?:回答|解答|确定|提供)",
    r"没有(?:找到|相关|足够)(?:的)?(?:资料|信息|依据|内容|政策)",
    r"资料(?:中|里)?(?:未|没有)(?:提及|包含|说明)",
    r"抱歉[，,].{0,20}(?:无法|不能|没有)",
    r"不(?:能|可)确定",
    r"未(?:能)?找到",
    r"暂无(?:相关)?(?:信息|资料|政策)",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS))

RAG_SYSTEM_PROMPT = """你是「拾汇商城」的官方智能客服助手，负责依据平台政策资料回答用户咨询。

【硬性规则】
1. 你只能依据下方【政策资料】中的内容作答，严禁使用你自己的先验知识、常识或推测补充任何平台政策。
2. 如果【政策资料】中没有足以回答该问题的信息，你必须明确拒答，回复格式为：
   「抱歉，我在平台政策知识库中没有找到与该问题相关的资料，因此无法给出准确回答。」
   然后可以建议用户补充信息或联系人工客服。绝对不可以猜测或编造。
3. 严禁编造政策条款编号、时间期限、金额比例、生效日期、适用条件等任何具体细节。
   资料中没有写的细节，就不要说。
4. 如果资料中的内容彼此冲突或版本不一致，请指出存在不同说法，并提示用户以最新官方公告为准。
5. 回答中涉及事实的部分，用 [编号] 标注依据的资料片段（例如：[1]、[2]），编号对应【政策资料】里的序号。
6. 使用简体中文，语气礼貌、专业、简洁；先给结论，再给要点。
7. 不要提及「检索」「向量」「上下文」「资料片段」等实现细节，直接以客服口吻回答。
8. 本助手只处理平台政策咨询；若用户要投诉、举报或反馈订单问题，应引导其说明具体情况以便用户自行创建投诉工单。

【政策资料】
{context}
"""

RAG_USER_PROMPT = "用户问题：{question}"


@dataclass
class RagAnswer:
    """RAG 问答的统一返回结构。"""

    question: str
    answer: str
    answerable: bool
    sources: list[dict[str, Any]] = field(default_factory=list)
    used_chunks: int = 0
    retrieval_count: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "answerable": self.answerable,
            "sources": self.sources,
            "used_chunks": self.used_chunks,
            "retrieval_count": self.retrieval_count,
            "reason": self.reason,
        }


def document_to_source(doc: Document, *, score: float | None = None) -> dict[str, Any]:
    """把命中的片段转成可展示的引用信息。"""
    metadata = doc.metadata or {}
    snippet = doc.page_content.strip().replace("\n", " ")
    return {
        "chunk_id": metadata.get("chunk_id", ""),
        "source": metadata.get("source", ""),
        "section": metadata.get("section") or metadata.get("title") or "",
        "doc_type": metadata.get("doc_type", ""),
        "score": round(float(score), 4) if score is not None else None,
        "snippet": snippet[:200] + ("…" if len(snippet) > 200 else ""),
    }


def looks_like_refusal(text: str) -> bool:
    """判断模型输出是否属于拒答。"""
    if not text or not text.strip():
        return True
    return bool(_REFUSAL_RE.search(text))


@lru_cache(maxsize=1)
def get_hybrid_retriever_cached() -> HybridPolicyRetriever:
    return HybridPolicyRetriever()


class PolicyRagChain:
    """政策问答链：混合检索 -> 相关性门控 -> 受约束生成。"""

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        retriever: HybridPolicyRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self._llm = llm
        self._retriever = retriever

    # ------------------------------------------------------------------
    @property
    def retriever(self) -> HybridPolicyRetriever:
        if self._retriever is None:
            self._retriever = get_hybrid_retriever_cached()
        return self._retriever

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            from app.agent.llm import build_chat_model

            self._llm = build_chat_model(self.settings)
        return self._llm

    # ------------------------------------------------------------------
    def retrieve_scored(self, question: str) -> list[tuple[Document, float]]:
        """混合检索并附带向量相似度分数。"""
        return self.retriever.retrieve_with_scores(question)

    def _gate(
        self,
        question: str,
        scored: list[tuple[Document, float]],
    ) -> tuple[list[tuple[Document, float]], RagAnswer | None]:
        """相关性门控：决定是「有依据可答」还是「直接拒答」。

        Returns:
            ``(通过门控的片段, 拒答结果或 None)``。拒答结果非空时调用方必须直接返回，
            不再调用大模型——既省成本，也从机制上杜绝无依据的幻觉。
        """
        if not scored:
            logger.info("检索无结果，直接拒答: %s", question[:60])
            return [], RagAnswer(
                question=question,
                answer=REFUSAL_NO_CONTEXT,
                answerable=False,
                reason="no_retrieval_result",
            )

        relevant = [
            (doc, score) for doc, score in scored if score >= self.settings.relevance_threshold
        ]
        if relevant:
            return relevant, None

        best = max(score for _, score in scored)
        logger.info(
            "全部片段低于阈值 %.2f（最高 %.4f），直接拒答: %s",
            self.settings.relevance_threshold,
            best,
            question[:60],
        )
        return [], RagAnswer(
            question=question,
            answer=REFUSAL_NO_CONTEXT,
            answerable=False,
            sources=[document_to_source(d, score=s) for d, s in scored[:3]],
            retrieval_count=len(scored),
            reason=f"below_threshold(best={best:.4f})",
        )

    def _build_messages(
        self,
        question: str,
        relevant: list[tuple[Document, float]],
    ) -> list[Any]:
        """构造受约束的生成消息。"""
        context = format_documents([doc for doc, _ in relevant])
        return [
            SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=RAG_USER_PROMPT.format(question=question)),
        ]

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> RagAnswer:
        """同步执行政策问答。"""
        question = (question or "").strip()
        if not question:
            return RagAnswer(
                question=question,
                answer="请描述您想咨询的平台政策问题。",
                answerable=False,
                reason="empty_question",
            )

        scored = self.retrieve_scored(question)[: top_k or self.settings.fusion_top_k]

        # --- 防线 1：相关性门控 ---
        relevant, refusal = self._gate(question, scored)
        if refusal is not None:
            return refusal

        # --- 防线 2：受约束生成 ---
        llm = self._get_llm()
        try:
            response = llm.invoke(self._build_messages(question, relevant))
        except Exception as exc:  # noqa: BLE001 - LLM 故障需给出可读降级信息
            logger.error("调用大模型失败: %s", exc)
            return RagAnswer(
                question=question,
                answer=(
                    "抱歉，智能客服当前暂时不可用（大模型调用失败），"
                    "请稍后重试或联系人工客服。"
                ),
                answerable=False,
                sources=[document_to_source(d, score=s) for d, s in relevant],
                used_chunks=len(relevant),
                retrieval_count=len(scored),
                reason=f"llm_error:{type(exc).__name__}",
            )

        answer_text = _extract_text(response)

        # --- 防线 3：拒答识别与统一契约 ---
        refused = looks_like_refusal(answer_text)
        if refused and not answer_text.strip():
            answer_text = REFUSAL_NO_CONTEXT

        return RagAnswer(
            question=question,
            answer=answer_text,
            answerable=not refused,
            sources=[document_to_source(d, score=s) for d, s in relevant],
            used_chunks=len(relevant),
            retrieval_count=len(scored),
            reason="refused_by_model" if refused else "ok",
        )

    async def aanswer(self, question: str, *, top_k: int | None = None) -> RagAnswer:
        """异步执行政策问答。

        与同步路径共用同一套门控与 Prompt 构造逻辑，保证行为一致：
        无关问题同样在调用大模型之前就拒答。
        """
        question = (question or "").strip()
        if not question:
            return RagAnswer(
                question=question,
                answer="请描述您想咨询的平台政策问题。",
                answerable=False,
                reason="empty_question",
            )

        limit = top_k or self.settings.fusion_top_k
        result = await self.retriever.aretrieve(question, top_k=limit)
        # EnsembleRetriever 只返回文档，这里补上向量分数以复用门控逻辑
        scored = await asyncio.to_thread(
            self.retriever.score_documents, question, result.documents, top_k=limit
        )

        relevant, refusal = self._gate(question, scored)
        if refusal is not None:
            return refusal

        try:
            response = await self._get_llm().ainvoke(self._build_messages(question, relevant))
        except Exception as exc:  # noqa: BLE001
            logger.error("异步调用大模型失败: %s", exc)
            return RagAnswer(
                question=question,
                answer="抱歉，智能客服当前暂时不可用（大模型调用失败），请稍后重试。",
                answerable=False,
                sources=[document_to_source(d, score=s) for d, s in relevant],
                used_chunks=len(relevant),
                reason=f"llm_error:{type(exc).__name__}",
            )

        answer_text = _extract_text(response)
        refused = looks_like_refusal(answer_text)
        return RagAnswer(
            question=question,
            answer=answer_text,
            answerable=not refused,
            sources=[document_to_source(d, score=s) for d, s in relevant],
            used_chunks=len(relevant),
            retrieval_count=len(scored),
            reason="refused_by_model" if refused else "ok",
        )


def _extract_text(response: Any) -> str:
    """从 LangChain 消息对象中取出纯文本。

    兼容 ``content`` 为字符串或内容块列表（部分模型返回 list[dict]）的两种形态。
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


@lru_cache(maxsize=1)
def get_rag_chain() -> PolicyRagChain:
    """全局唯一 RAG 链。"""
    return PolicyRagChain()
