"""不依赖外部服务（Milvus / 大模型 / 向量模型）的单元测试。

运行::

    python -m pytest tests/test_units.py -v
    或
    python tests/test_units.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.request_context import (  # noqa: E402
    JWT_CONFIG_KEY,
    MissingJwtContextError,
    build_run_config,
    get_jwt_from_config,
    get_session_id_from_config,
)
from app.rag.hybrid_retriever import _reciprocal_rank_fusion, format_documents  # noqa: E402
from app.rag.loader import (  # noqa: E402
    chunk_document,
    deduplicate,
    load_document,
    split_into_sections,
)
from app.rag.rag_chain import looks_like_refusal  # noqa: E402
from app.utils.jwt import (  # noqa: E402
    MissingTokenError,
    extract_bearer_token,
    token_fingerprint,
)
from app.utils.tokenizer import tokenize  # noqa: E402

from langchain_core.documents import Document  # noqa: E402


# ----------------------------------------------------------------------
# 中文分词
# ----------------------------------------------------------------------
def test_tokenize_chinese_splits_words() -> None:
    """中文必须被切成词，而不是整句一个 token。"""
    tokens = tokenize("七天无理由退货的运费由谁承担")
    assert len(tokens) > 1, f"分词失败，仅得到 {tokens}"
    assert any("退货" in token for token in tokens)
    assert any("运费" in token for token in tokens)


def test_tokenize_filters_stopwords_and_punctuation() -> None:
    tokens = tokenize("请问，退货的运费是多少？")
    assert "的" not in tokens
    assert "，" not in tokens
    assert "？" not in tokens
    assert "请问" in tokens


def test_tokenize_empty_input() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []


# ----------------------------------------------------------------------
# JWT 提取
# ----------------------------------------------------------------------
def test_extract_bearer_token_ok() -> None:
    token = extract_bearer_token("Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123")
    assert token == "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123"


def test_extract_bearer_token_case_insensitive() -> None:
    assert extract_bearer_token("bearer abcdefgh12345") == "abcdefgh12345"


def test_extract_bearer_token_rejects_missing() -> None:
    for bad in (None, "", "   "):
        try:
            extract_bearer_token(bad)
        except MissingTokenError:
            continue
        raise AssertionError(f"应当拒绝: {bad!r}")


def test_extract_bearer_token_rejects_wrong_scheme() -> None:
    for bad in ("Basic abcdefgh12345", "eyJhbGciOiJIUzI1NiJ9.payload.sig", "Bearer  short"):
        try:
            extract_bearer_token(bad)
        except MissingTokenError:
            continue
        raise AssertionError(f"应当拒绝: {bad!r}")


def test_token_fingerprint_is_stable_and_short() -> None:
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123"
    first = token_fingerprint(token)
    assert first == token_fingerprint(token)
    assert first.startswith("sha256:")
    assert token not in first
    assert token_fingerprint("") == "<empty>"


# ----------------------------------------------------------------------
# RunnableConfig 透传
# ----------------------------------------------------------------------
def test_build_run_config_carries_jwt() -> None:
    config = build_run_config("tok-123", session_id="s1", user_tag="u1")
    assert config["configurable"][JWT_CONFIG_KEY] == "tok-123"
    assert get_jwt_from_config(config) == "tok-123"
    assert get_session_id_from_config(config) == "s1"


def test_get_jwt_from_config_reads_nested_and_flat() -> None:
    """LangGraph 在不同版本里把键放在不同层级，两种都要能读到。"""
    assert get_jwt_from_config({"configurable": {JWT_CONFIG_KEY: "nested"}}) == "nested"
    assert get_jwt_from_config({JWT_CONFIG_KEY: "flat"}) == "flat"
    assert get_jwt_from_config({"metadata": {JWT_CONFIG_KEY: "meta"}}) == "meta"


def test_get_jwt_from_config_raises_when_missing() -> None:
    for bad in (None, {}, {"configurable": {}}, {"configurable": {JWT_CONFIG_KEY: "  "}}):
        try:
            get_jwt_from_config(bad)
        except MissingJwtContextError:
            continue
        raise AssertionError(f"应当抛出 MissingJwtContextError: {bad!r}")


# ----------------------------------------------------------------------
# 文档分块
# ----------------------------------------------------------------------
def test_split_into_sections_by_markdown_heading() -> None:
    text = "# 标题\n\n## 一、适用范围\n内容A\n\n## 二、退货时限\n内容B\n"
    sections = split_into_sections(text)
    titles = [title for title, _ in sections]
    assert any("适用范围" in title for title in titles)
    assert any("退货时限" in title for title in titles)


def test_split_into_sections_by_chinese_heading() -> None:
    text = "第一条 适用范围\n内容A\n\n第二条 退货时限\n内容B\n"
    sections = split_into_sections(text)
    titles = [title for title, _ in sections]
    assert any("第一条" in title for title in titles)
    assert any("第二条" in title for title in titles)


def test_chunk_document_produces_metadata() -> None:
    policy = PROJECT_ROOT / "data" / "policies" / "退换货政策.md"
    if not policy.exists():
        return

    loaded = load_document(policy)
    chunks = chunk_document(loaded)
    assert chunks, "应当产生至少一个片段"

    for chunk in chunks:
        metadata = chunk.metadata
        assert metadata["chunk_id"]
        assert metadata["source"] == "退换货政策.md"
        assert metadata["doc_type"] == "md"
        assert metadata["section"]
        assert isinstance(metadata["chunk_index"], int)

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 必须唯一"


def test_chunk_document_respects_chunk_size() -> None:
    policy = PROJECT_ROOT / "data" / "policies" / "退换货政策.md"
    if not policy.exists():
        return
    loaded = load_document(policy)
    chunks = chunk_document(loaded)
    # 允许少量超出（递归切分在无合适分隔符时会切在硬边界）
    oversized = [c for c in chunks if len(c.page_content) > 500 * 1.6]
    assert not oversized, f"存在明显超长片段: {[len(c.page_content) for c in oversized]}"


def test_deduplicate_removes_repeats() -> None:
    docs = [
        Document(page_content="a", metadata={"chunk_id": "1"}),
        Document(page_content="a", metadata={"chunk_id": "1"}),
        Document(page_content="b", metadata={"chunk_id": "2"}),
    ]
    assert len(deduplicate(docs)) == 2


def test_load_document_rejects_unsupported_suffix() -> None:
    bad = PROJECT_ROOT / "data" / "policies" / "test.unsupported"
    bad.write_text("x", encoding="utf-8")
    try:
        load_document(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("不支持的格式应当抛 ValueError")
    finally:
        bad.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# RRF 兜底实现
# ----------------------------------------------------------------------
def test_reciprocal_rank_fusion_prefers_documents_hit_by_both() -> None:
    dense = [
        Document(page_content="only-dense", metadata={"chunk_id": "d1"}),
        Document(page_content="shared", metadata={"chunk_id": "s"}),
    ]
    sparse = [
        Document(page_content="shared", metadata={"chunk_id": "s"}),
        Document(page_content="only-sparse", metadata={"chunk_id": "k1"}),
    ]
    fused = _reciprocal_rank_fusion([dense, sparse], [0.5, 0.5], c=60)
    assert fused[0].metadata["chunk_id"] == "s", "两路都命中的文档应排第一"
    assert len(fused) == 3, "融合结果应去重"


def test_reciprocal_rank_fusion_applies_weights() -> None:
    dense = [Document(page_content="a", metadata={"chunk_id": "a"})]
    sparse = [Document(page_content="b", metadata={"chunk_id": "b"})]
    fused = _reciprocal_rank_fusion([dense, sparse], [0.9, 0.1], c=60)
    assert fused[0].metadata["chunk_id"] == "a", "权重高的一路应排前"


# ----------------------------------------------------------------------
# 拒答识别与上下文格式化
# ----------------------------------------------------------------------
def test_looks_like_refusal_detects_refusals() -> None:
    refusals = [
        "抱歉，我在平台政策知识库中没有找到与该问题相关的资料，因此无法给出准确回答。",
        "资料中未提及该情形。",
        "我无法回答这个问题。",
        "",
    ]
    for text in refusals:
        assert looks_like_refusal(text), f"应识别为拒答: {text!r}"


def test_looks_like_refusal_allows_real_answers() -> None:
    answers = [
        "七天无理由退货的时限是签收之日起 7 个自然日 [1]。",
        "退货运费由商家承担，前提是商品存在质量问题 [2]。",
    ]
    for text in answers:
        assert not looks_like_refusal(text), f"不应误判为拒答: {text!r}"


def test_format_documents_numbers_and_labels() -> None:
    docs = [
        Document(
            page_content="内容一",
            metadata={"source": "a.md", "section": "一、适用范围"},
        ),
        Document(page_content="内容二", metadata={"source": "b.md", "section": "二、时限"}),
    ]
    text = format_documents(docs)
    assert "[1]" in text and "[2]" in text
    assert "a.md" in text and "一、适用范围" in text
    assert text.index("[1]") < text.index("[2]")


def test_format_documents_truncates_long_content() -> None:
    docs = [Document(page_content="字" * 5000, metadata={"source": "a.md"})]
    text = format_documents(docs, max_chars=100)
    assert "…" in text
    assert len(text) < 400


# ----------------------------------------------------------------------
# 消息文本提取
# ----------------------------------------------------------------------
def test_extract_text_from_message_handles_block_content() -> None:
    from langchain_core.messages import AIMessage

    from app.rag.rag_chain import _extract_text

    assert _extract_text(AIMessage(content="纯文本")) == "纯文本"
    assert _extract_text(AIMessage(content=[{"type": "text", "text": "块文本"}])) == "块文本"
    assert (
        _extract_text(AIMessage(content=[{"text": "a"}, "b"])) == "ab"
    )


def _run_all() -> int:
    """不依赖 pytest 的简易运行器。"""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures: list[str] = []
    for name, func in tests:
        try:
            func()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")

    print("-" * 60)
    print(f"共 {len(tests)} 个用例，失败 {len(failures)} 个")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
