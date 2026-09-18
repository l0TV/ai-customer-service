"""相关性阈值标定：量化「相关问题」与「无关问题」的相似度分布。

用途：``AI_CS_RELEVANCE_THRESHOLD`` 是拒答机制的第一道防线，
但 bge-small-zh 的 COSINE 分值区间较窄，语料变化时会漂移，
需要用真实语料重新标定，而不是沿用旧值。

运行::

    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402

# 应当能答上来的问题（命中知识库）
RELEVANT_QUERIES = [
    "七天无理由退货是几天",
    "退货运费谁承担",
    "运费多少钱包邮",
    "发票开错了能重开吗",
    "退款多久到账",
    "VIP 等级怎么升级",
    "积分怎么抵扣",
    "优惠券能叠加吗",
    "价保多少天",
    "签收后发现破损怎么办",
]

# 应当拒答的问题（平台政策知识库不可能覆盖）
IRRELEVANT_QUERIES = [
    "如何用 Python 训练卷积神经网络识别猫狗图片",
    "帮我写一首关于秋天的七言绝句",
    "证明费马大定理的简要思路",
    "今天北京的天气怎么样",
    "推荐几部好看的科幻电影",
    "量子纠缠的物理原理是什么",
    "怎么做红烧肉",
    "C++ 的虚函数表是怎么实现的",
]


def main() -> int:
    setup_logging("WARNING")

    from app.rag.hybrid_retriever import get_hybrid_retriever_cached

    retriever = get_hybrid_retriever_cached()
    retriever.ensure_ready()

    print(f"语料规模      : {retriever.corpus_size} 个片段")
    print(f"当前阈值      : {settings.relevance_threshold}")
    print(f"两路召回 top-k: dense={settings.dense_top_k}, sparse={settings.sparse_top_k}")
    print(f"融合后条数    : {settings.fusion_top_k}")
    print()

    def top_score(query: str) -> tuple[float, str]:
        scored = retriever.retrieve_with_scores(query)
        if not scored:
            return 0.0, "(无结果)"
        doc, score = scored[0]
        return float(score), str(doc.metadata.get("source", "?"))

    print("=" * 78)
    print("应能回答的问题")
    print("=" * 78)
    relevant_scores: list[float] = []
    for query in RELEVANT_QUERIES:
        score, source = top_score(query)
        relevant_scores.append(score)
        flag = "OK " if score >= settings.relevance_threshold else "会被拒答!"
        print(f"  {score:.4f}  {flag}  {query:<28} <- {source}")

    print()
    print("=" * 78)
    print("应当拒答的问题")
    print("=" * 78)
    irrelevant_scores: list[float] = []
    for query in IRRELEVANT_QUERIES:
        score, source = top_score(query)
        irrelevant_scores.append(score)
        flag = "会拒答 OK" if score < settings.relevance_threshold else "漏放(会作答)!"
        print(f"  {score:.4f}  {flag}  {query:<28} <- {source}")

    relevant_scores.sort()
    irrelevant_scores.sort()

    print()
    print("=" * 78)
    print("分布汇总")
    print("=" * 78)
    print(f"  相关问题: 最低={relevant_scores[0]:.4f}  中位={relevant_scores[len(relevant_scores)//2]:.4f}  最高={relevant_scores[-1]:.4f}")
    print(f"  无关问题: 最低={irrelevant_scores[0]:.4f}  中位={irrelevant_scores[len(irrelevant_scores)//2]:.4f}  最高={irrelevant_scores[-1]:.4f}")
    print()
    print(f"  无关问题最高分 = {irrelevant_scores[-1]:.4f}")
    print(f"  相关问题最低分 = {relevant_scores[0]:.4f}")

    gap_low, gap_high = irrelevant_scores[-1], relevant_scores[0]
    if gap_low < gap_high:
        mid = (gap_low + gap_high) / 2
        print(f"  存在清晰间隔 [{gap_low:.4f}, {gap_high:.4f}]，建议阈值取中点 ≈ {mid:.2f}")
    else:
        print(f"  ⚠ 两类分布重叠（无关最高 {gap_low:.4f} >= 相关最低 {gap_high:.4f}）")
        print("    单靠绝对阈值无法完全区分，需要：")
        print("      a) 补充/收紧语料，减少通用性过强的泛化段落；")
        print("      b) 用更强的向量模型（如 bge-base-zh / bge-m3）；")
        print("      c) 依赖生成阶段的 Prompt 约束兜底拒答。")

    failed_relevant = [q for q, s in zip(RELEVANT_QUERIES, [top_score(q)[0] for q in RELEVANT_QUERIES])
                       if s < settings.relevance_threshold]
    leaked = [q for q, s in zip(IRRELEVANT_QUERIES, [top_score(q)[0] for q in IRRELEVANT_QUERIES])
              if s >= settings.relevance_threshold]

    print()
    print(f"  当前阈值下会被误拒的相关问题: {len(failed_relevant)} 个 {failed_relevant}")
    print(f"  当前阈值下会漏放的无关问题  : {len(leaked)} 个 {leaked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
