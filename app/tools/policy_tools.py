"""政策检索工具：把混合检索（BM25 + 向量 + RRF）暴露给 Agent。

Agent 自主决策的核心就在这里——当它判断用户问的是政策类问题，
就会调用本工具去知识库检索，然后严格基于检索结果作答。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.config import settings as default_settings
from app.core.logging_config import get_logger
from app.rag.hybrid_retriever import format_documents
from app.rag.rag_chain import REFUSAL_NO_CONTEXT, get_hybrid_retriever_cached

logger = get_logger(__name__)


class PolicySearchInput(BaseModel):
    """政策检索工具入参。"""

    query: str = Field(
        description=(
            "要在平台政策知识库中检索的问题或关键词，使用简体中文。"
            "建议保留用户问题中的关键实体（如「七天无理由退货」「运费」「发票」），"
            "不要加入寒暄语。"
        ),
        min_length=1,
        max_length=500,
    )
    top_k: int = Field(
        default=0,
        description="需要返回的片段数量，0 表示使用系统默认值（推荐保持 0）。",
        ge=0,
        le=20,
    )


@tool("search_platform_policy", args_schema=PolicySearchInput)
def search_platform_policy(query: str, top_k: int = 0) -> str:
    """检索拾汇商城的平台政策知识库（用户协议与账户、订单与交易、支付、发票、促销、会员、优惠券、
    积分、配送与签收、运费、退换货、退款、价格保护、评价、售后等）。

    当用户咨询平台规则、政策、流程、时限、费用标准等「规定类」问题时调用本工具。
    返回的是知识库中的原文片段，你必须严格依据这些片段回答，
    如果片段中没有答案就明确告知无法回答，不得凭自身知识补充。

    Args:
        query: 检索问题。
        top_k: 返回片段数，0 表示默认。

    Returns:
        原文片段文本；无结果时返回明确的空结果提示。
    """
    query = (query or "").strip()
    if not query:
        return "检索问题为空，请重新描述您的政策咨询内容。"

    limit = top_k or default_settings.fusion_top_k
    scored = get_hybrid_retriever_cached().retrieve_with_scores(query)[:limit]

    if not scored:
        logger.info("政策检索无结果: %s", query[:80])
        return (
            "【检索结果】知识库中没有检索到任何相关片段。\n"
            f"请据此回复用户：{REFUSAL_NO_CONTEXT}"
        )

    # 只把达到相关性阈值的片段交给模型，避免用弱相关内容诱导幻觉
    relevant = [
        (doc, score) for doc, score in scored if score >= default_settings.relevance_threshold
    ]
    if not relevant:
        best = max(score for _, score in scored)
        logger.info("政策检索全部低于阈值 (best=%.4f): %s", best, query[:80])
        return (
            f"【检索结果】找到 {len(scored)} 个片段，但相关度均低于阈值"
            f"（最高相似度 {best:.3f}），不足以作为回答依据。\n"
            f"请据此回复用户：{REFUSAL_NO_CONTEXT}"
        )

    context = format_documents([doc for doc, _ in relevant])
    logger.info("政策检索命中 %d/%d 个片段: %s", len(relevant), len(scored), query[:80])
    return (
        f"【检索结果】以下 {len(relevant)} 个片段来自平台政策知识库，"
        "是你的唯一作答依据：\n\n"
        f"{context}\n\n"
        "【作答要求】仅依据上述片段回答；片段不足以回答时必须明确拒答并建议联系人工客服"
    )


def get_policy_tools() -> list[Any]:
    """返回政策类工具列表。"""
    return [search_platform_policy]
