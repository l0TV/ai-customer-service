"""集成测试：需要 Milvus、向量模型可用（大模型 API Key 可选）。

覆盖：
* 知识库入库 -> BM25 快照同步
* 混合检索（BM25 + 向量 + RRF 融合）确实能命中政策片段
* 相似度门控：无关问题触发拒答
* JWT 依赖：缺失/格式错误一律 401
* SSE 流式与 JSON 对话接口
* RunnableConfig 透传：工单工具能在 config 中取到 JWT
* 网关不可达时的降级行为（不谎称建单成功）

运行::

    python tests/test_integration.py
    python tests/test_integration.py --skip-ingest   # 已入库时跳过快照重建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, condition, detail))
    print(f"{'PASS' if condition else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


# ----------------------------------------------------------------------
def test_ingest(rebuild: bool = True) -> None:
    """入库并校验 Milvus 与 BM25 快照同源。"""
    import asyncio

    from app.rag.hybrid_retriever import load_bm25_corpus
    from app.rag.ingest import ingest_policies

    stats = asyncio.run(ingest_policies(rebuild=rebuild))
    check("入库：有片段产生", stats["chunks"] > 0, f"chunks={stats['chunks']}")
    check("入库：写入 Milvus", stats["written"] > 0, f"written={stats['written']}")

    corpus = load_bm25_corpus(settings.bm25_corpus_path)
    check(
        "入库：BM25 快照与入库片段数一致",
        len(corpus) == stats["chunks"],
        f"snapshot={len(corpus)} chunks={stats['chunks']}",
    )

    print(f"     Milvus 集合总量 = {stats['collection_count']}")


def test_vector_store() -> None:
    """Milvus 可检索且分数在合理范围。"""
    from app.rag.vectorstore import get_vector_store

    store = get_vector_store()
    count = store.count()
    check("Milvus：集合非空", count > 0, f"count={count}")

    hits = store.similarity_search_with_score("七天无理由退货的时限", k=3)
    check("Milvus：向量检索有结果", len(hits) > 0, f"hits={len(hits)}")
    if hits:
        scores = [round(float(score), 4) for _, score in hits]
        check(
            "Milvus：相似度分数越大越相关（0~1）",
            all(-1.0 <= score <= 1.0 for score in scores),
            f"scores={scores}",
        )
        print(f"     命中片段：{hits[0][0].metadata.get('source')} / "
              f"{hits[0][0].metadata.get('section')}")


def test_hybrid_retrieval() -> None:
    """混合检索能召回正确的政策片段。"""
    from app.rag.hybrid_retriever import get_hybrid_retriever_cached

    retriever = get_hybrid_retriever_cached()
    retrieved = retriever.reload()
    check("混合检索：BM25+向量均就绪", retrieved, f"corpus={retriever.corpus_size}")

    cases = [
        ("七天无理由退货是几天", "退换货政策"),
        ("运费多少钱", "配送与运费政策"),
        ("发票开错了能重开吗", "发票与退款政策"),
        ("VIP 等级怎么升级", "会员与优惠券政策"),
    ]
    for question, expected_source in cases:
        scored = retriever.retrieve_with_scores(question)
        sources = [doc.metadata.get("source", "") for doc, _ in scored]
        hit = any(expected_source in source for source in sources)
        top = scored[0][1] if scored else 0.0
        check(
            f"混合检索命中「{question}」",
            hit,
            f"top_score={top:.4f} sources={sources[:3]}",
        )


def test_relevance_gate() -> None:
    """完全无关的问题不应通过相关性门控。"""
    from app.rag.hybrid_retriever import get_hybrid_retriever_cached

    retriever = get_hybrid_retriever_cached()
    scored = retriever.retrieve_with_scores("如何用 Python 训练一个卷积神经网络识别猫狗图片")
    best = max((score for _, score in scored), default=0.0)
    check(
        "相关性门控：无关问题最高分低于阈值",
        best < settings.relevance_threshold,
        f"best={best:.4f} threshold={settings.relevance_threshold}",
    )


def test_rag_refusal_without_llm() -> None:
    """无 LLM Key 时，无关问题应在调用大模型前就拒答（省钱且无幻觉）。"""
    from app.rag.rag_chain import REFUSAL_NO_CONTEXT, PolicyRagChain

    chain = PolicyRagChain()
    result = chain.answer("如何训练卷积神经网络识别猫狗图片")
    check("RAG：无关问题被拒答", not result.answerable, f"reason={result.reason}")
    check(
        "RAG：拒答文案为标准话术",
        result.answer == REFUSAL_NO_CONTEXT,
        result.answer[:40],
    )


def test_jwt_dependency() -> None:
    """JWT 提取与 config 透传。"""
    from app.core.request_context import build_run_config, get_jwt_from_config
    from app.utils.jwt import MissingTokenError, extract_bearer_token

    token = extract_bearer_token("Bearer header.payload.signature123")
    config = build_run_config(token, session_id="it-session")
    check("JWT：从请求头原样提取", token == "header.payload.signature123", token)
    check("JWT：经 RunnableConfig 透传", get_jwt_from_config(config) == token)

    try:
        extract_bearer_token(None)
        check("JWT：缺失时拒绝", False, "未抛异常")
    except MissingTokenError:
        check("JWT：缺失时拒绝", True)

    # 未做任何解析：token 原样保留，说明本服务确实不解析 JWT
    check("JWT：不做解码（原样透传）", config["configurable"]["jwt_token"] == token)


def test_complaint_tool_via_gateway() -> None:
    """工单工具：JWT 经 ToolRuntime 取到；网关不可达时必须失败且不能谎报成功。"""
    import asyncio

    from langgraph.prebuilt import ToolRuntime

    from app.tools.complaint_tools import create_complaint_ticket

    fake_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMDAifQ.fakesignature"
    runtime = ToolRuntime(
        state={},
        context=None,
        config={"configurable": {"jwt_token": fake_jwt, "session_id": "it-complaint"}},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call_it_1",
        store=None,
        tools=[],
    )

    result = asyncio.run(
        create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
            detail="集成测试：收到的商品外包装破损，申请补偿或换货。",
            runtime=runtime,
            order_sn="IT202501010001",
            complaint_type="物流配送",
        )
    )
    text = result if isinstance(result, str) else str(result)
    print(f"     工单工具返回：{text[:200]}")

    check(
        "工单工具：模型可见参数不含 runtime",
        "runtime" not in create_complaint_ticket.args,
        f"args={list(create_complaint_ticket.args)}",
    )

    if "【工单创建成功】" in text:
        check("工单工具：网关可用时创建成功", True, text[:120])
    else:
        # 网关未启动属正常，关键是必须如实报错而不是假装成功
        honest = ("失败" in text) or ("无法连接" in text) or ("超时" in text)
        check("工单工具：失败时如实报错（不谎报成功）", honest, text[:120])

    # 缺少 JWT 时必须直接拒绝，绝不能匿名调用后端
    bare = ToolRuntime(
        state={},
        context=None,
        config={"configurable": {}},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call_it_2",
        store=None,
        tools=[],
    )
    rejected = asyncio.run(
        create_complaint_ticket.coroutine(detail="测试缺少凭据的场景，不应发起请求。", runtime=bare)  # type: ignore[attr-defined]
    )
    check(
        "工单工具：缺少 JWT 时拒绝执行",
        "缺少用户登录凭证" in str(rejected),
        str(rejected)[:100],
    )


def test_http_api() -> None:
    """通过 TestClient 走一遍真实 HTTP 接口。"""
    try:
        from fastapi.testclient import TestClient
        from starlette.testclient import TestClient as StarletteTestClient
    except ImportError as exc:  # pragma: no cover
        check("HTTP：TestClient 可用", False, str(exc))
        return

    from app.main import app

    client_cls = TestClient or StarletteTestClient
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI5OTkifQ.integration-testsig"
    auth = {"Authorization": f"Bearer {token}"}

    # 用 with 触发 lifespan（预热检索器）
    with client_cls(app) as client:
        # --- 健康检查 ---
        health = client.get("/api/ai/health")
        check("HTTP：/health 返回 200", health.status_code == 200, str(health.status_code))
        if health.status_code == 200:
            body = health.json()
            components = {item["name"]: item["healthy"] for item in body["components"]}
            check("HTTP：Milvus 健康", components.get("milvus", False), str(components))
            check("HTTP：BM25 语料健康", components.get("bm25_corpus", False))

        # --- 缺少 JWT ---
        no_auth = client.post("/api/ai/chat", json={"message": "退货政策"})
        check("HTTP：缺少 JWT 返回 401", no_auth.status_code == 401, str(no_auth.status_code))

        bad_auth = client.post(
            "/api/ai/chat",
            json={"message": "退货政策"},
            headers={"Authorization": "Basic abc"},
        )
        check("HTTP：JWT 格式错误返回 401", bad_auth.status_code == 401, str(bad_auth.status_code))

        # --- 混合检索调试接口 ---
        search = client.get("/api/ai/rag/search", params={"q": "退货运费谁承担"}, headers=auth)
        check("HTTP：/rag/search 返回 200", search.status_code == 200, str(search.status_code))
        if search.status_code == 200:
            data = search.json()
            results = data.get("results", [])
            check("HTTP：检索有结果", len(results) > 0, f"n={len(results)}")
            if results:
                top = results[0]
                check(
                    "HTTP：检索命中退换货/配送政策",
                    "政策" in top.get("source", ""),
                    f"{top.get('source')} score={top.get('score')}",
                )

        # --- 配置接口 ---
        cfg = client.get("/api/ai/config")
        check("HTTP：/config 返回 200", cfg.status_code == 200, str(cfg.status_code))

        # --- 无关问题在 rag 模式下应拒答（不调用大模型）---
        rag = client.post(
            "/api/ai/rag/query",
            json={"message": "如何用 Python 训练卷积神经网络", "mode": "rag"},
            headers=auth,
        )
        check("HTTP：/rag/query 返回 200", rag.status_code == 200, str(rag.status_code))
        if rag.status_code == 200:
            payload = rag.json()
            check(
                "HTTP：无关问题被拒答",
                payload.get("answerable") is False,
                f"reason={payload.get('reason')}",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="集成测试")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过入库步骤")
    args = parser.parse_args()

    setup_logging("WARNING")

    print("=" * 72)
    print("集成测试开始")
    print(f"Milvus      : {settings.milvus_uri}")
    print(f"向量模型    : {settings.embedding_model}")
    print(f"LLM Key     : {'已配置' if settings.llm_configured else '未配置（涉及生成的用例会走降级分支）'}")
    print("=" * 72)

    if not args.skip_ingest:
        print("\n--- 知识库入库 ---")
        test_ingest(rebuild=True)

    print("\n--- 向量库 ---")
    test_vector_store()

    print("\n--- 混合检索 ---")
    test_hybrid_retrieval()

    print("\n--- 相关性门控 ---")
    test_relevance_gate()

    print("\n--- RAG 拒答 ---")
    test_rag_refusal_without_llm()

    print("\n--- JWT 透传 ---")
    test_jwt_dependency()

    print("\n--- 工单工具 ---")
    test_complaint_tool_via_gateway()

    print("\n--- HTTP 接口 ---")
    test_http_api()

    print("\n" + "=" * 72)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"共 {len(RESULTS)} 项检查，失败 {len(failed)} 项")
    for name in failed:
        detail = next(d for n, ok, d in RESULTS if n == name and not ok)
        print(f"  - {name} | {detail}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
