"""混合检索：BM25 关键词 + 向量语义，使用 RRF（倒数排名融合）合并。

实现要点
--------
1. **BM25 侧**：直接使用 LangChain 的 ``BM25Retriever``，但必须传入中文分词
   函数（``preprocess_func``），否则默认的英文空白分词会把整句中文当成一个
   token，关键词召回形同失效。
2. **向量侧**：``PolicyVectorStore.as_retriever()`` 返回 Milvus 相似度检索器。
3. **融合**：优先使用 LangChain 的 ``EnsembleRetriever``。它实现的就是加权
   RRF：``score(d) = Σ w_i / (rank_i(d) + c)``，``c`` 默认 60，并且会按
   ``id_key`` 对多路结果去重并累加分数——正是本项目需要的语义。
4. **语料持久化**：``BM25Retriever`` 必须在构造时拿到全量语料并在内存建索引，
   因此入库时把分块结果快照到 ``data/store/bm25_corpus.json``，服务启动直接
   载入，无需重新解析 PDF/Word。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger
from app.utils.tokenizer import bm25_preprocess

logger = get_logger(__name__)

# RRF 去重所用的元数据键（比 page_content 更稳，避免标点差异导致重复）
RRF_ID_KEY = "chunk_id"


# ----------------------------------------------------------------------
# BM25 语料快照
# ----------------------------------------------------------------------
def save_bm25_corpus(documents: Sequence[Document], path: Path | None = None) -> Path:
    """把分块结果落盘为 BM25 语料快照。"""
    target = path or default_settings.bm25_corpus_path
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents
    ]
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )
    logger.info("BM25 语料快照已保存: %s（%d 条）", target, len(payload))
    return target


def load_bm25_corpus(path: Path | None = None) -> list[Document]:
    """载入 BM25 语料快照；文件不存在时返回空列表。"""
    target = path or default_settings.bm25_corpus_path
    if not target.exists():
        logger.warning("BM25 语料快照不存在: %s（请先执行入库）", target)
        return []

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("BM25 语料快照损坏: %s", exc)
        return []

    documents = [
        Document(page_content=item.get("page_content", ""), metadata=item.get("metadata") or {})
        for item in raw
        if item.get("page_content")
    ]
    logger.info("已载入 BM25 语料 %d 条", len(documents))
    return documents


# ----------------------------------------------------------------------
# 检索器构建
# ----------------------------------------------------------------------
def build_bm25_retriever(
    documents: Sequence[Document],
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> BaseRetriever | None:
    """构建中文 BM25 检索器。"""
    cfg = settings or default_settings
    if not documents:
        logger.warning("BM25 语料为空，跳过 BM25 检索器构建")
        return None

    # langchain_community 已经 deprecated
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(
        list(documents),
        preprocess_func=bm25_preprocess,
        bm25_params={"k1": 1.5, "b": 0.75},
    )
    retriever.k = top_k or cfg.sparse_top_k
    logger.info("BM25 检索器就绪（语料 %d 条, k=%d）", len(documents), retriever.k)
    return retriever


def build_ensemble_retriever(
    retrievers: Sequence[BaseRetriever],
    weights: Sequence[float],
    *,
    c: int | None = None,
    settings: Settings | None = None,
) -> BaseRetriever:
    """用 LangChain ``EnsembleRetriever`` 做加权 RRF 融合。"""
    cfg = settings or default_settings

    from langchain_classic.retrievers import EnsembleRetriever

    ensemble = EnsembleRetriever(
        retrievers=list(retrievers),
        weights=list(weights),
        c=c if c is not None else cfg.rrf_c,
        id_key=RRF_ID_KEY,
    )
    logger.info(
        "EnsembleRetriever 就绪（RRF: c=%d, weights=%s, id_key=%s）",
        ensemble.c,
        list(weights),
        RRF_ID_KEY,
    )
    return ensemble


@dataclass
class RetrievalResult:
    """一次混合检索的结果。"""

    documents: list[Document]
    dense_documents: list[Document]
    sparse_documents: list[Document]
    query: str

    @property
    def is_empty(self) -> bool:
        return not self.documents


class HybridPolicyRetriever:
    """混合检索门面。

    对外只暴露 ``retrieve``，内部维护 BM25 + Milvus + EnsembleRetriever 三者，
    并保证语料变更后可热更新。

    线程安全说明：``reload`` 会整体替换内部引用，读操作先取本地引用再使用，
    因此并发读不会看到半成品状态。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._bm25: BaseRetriever | None = None
        self._ensemble: BaseRetriever | None = None
        self._corpus_size = 0
        # 保护 ensure_ready() 的懒加载，避免冷启动并发首请求重复构建索引
        self._reload_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reload(self) -> bool:
        """（重新）从快照与 Milvus 构建检索器。

        Returns:
            是否成功构建出混合检索器。BM25 语料为空时退化为纯向量检索。
        """
        from app.rag.vectorstore import get_vector_store

        documents = load_bm25_corpus(self.settings.bm25_corpus_path)
        self._corpus_size = len(documents)

        self._bm25 = build_bm25_retriever(
            documents, top_k=self.settings.sparse_top_k, settings=self.settings
        )

        dense = get_vector_store().as_retriever(k=self.settings.dense_top_k)

        if self._bm25 is None:
            # 没有 BM25 语料时仍提供向量检索，保证服务可用
            logger.warning("混合检索退化为纯向量检索（缺少 BM25 语料快照）")
            self._ensemble = dense
            return False

        self._ensemble = build_ensemble_retriever(
            [dense, self._bm25],
            [self.settings.rrf_weight_dense, self.settings.rrf_weight_sparse],
            c=self.settings.rrf_c,
            settings=self.settings,
        )
        return True

    def ensure_ready(self) -> None:
        """确保检索器已加载（懒加载 + 线程安全）。

        ``reload()`` 是重操作（读语料、连 Milvus、首次还会加载向量模型）。
        去掉启动时的强制加载后，冷启动瞬间的并发首请求可能同时进入这里，
        若不串行化就会重复构建多份索引并浪费十几秒。

        实现要点：
        * 快路径无锁（读完 ``_ensemble`` 非空即走），不影响正常请求性能；
        * 慢路径加锁并二次检查，保证同一时刻只有一次构建；
        * ``reload()`` 内部**最后**才给 ``_ensemble`` 赋值，因此读到非 None
          即隐含 ``_bm25``、``_corpus_size`` 均已写入，不会看到半成品状态。
        """
        if self._ensemble is not None:
            return

        with self._reload_lock:
            # 二次检查：可能已被其它线程或在锁外完成的 reload 填好
            if self._ensemble is not None:
                return
            self.reload()

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retrieve_dense: bool = True,
        retrieve_sparse: bool = True,
    ) -> RetrievalResult:
        """执行混合检索。

        Args:
            query: 用户问题。
            top_k: 融合后保留条数，默认取配置。
            retrieve_dense: 是否执行向量检索。
            retrieve_sparse: 是否执行 BM25 检索（调试对比时可关闭）。

        Returns:
            ``RetrievalResult``，含融合结果与两路原始结果（便于可观测性）。
        """
        self.ensure_ready()
        limit = top_k or self.settings.fusion_top_k

        dense_docs: list[Document] = []
        sparse_docs: list[Document] = []

        if retrieve_dense and retrieve_sparse and self._bm25 is not None:
            # 走 EnsembleRetriever：单次调用内部并发/串行执行两路并做 RRF
            fused = list(self._ensemble.invoke(query))  # type: ignore[union-attr]
            return RetrievalResult(
                documents=fused[:limit],
                dense_documents=[],
                sparse_documents=[],
                query=query,
            )

        if retrieve_dense:
            from app.rag.vectorstore import get_vector_store

            dense_docs = list(
                get_vector_store().as_retriever(k=self.settings.dense_top_k).invoke(query)
            )

        if retrieve_sparse:
            if self._bm25 is not None:
                sparse_docs = list(self._bm25.invoke(query))
            else:
                logger.warning("BM25 检索器不可用，本次仅返回向量结果")

        if retrieve_dense and retrieve_sparse:
            # 理论上不会到这里（上面已走 ensemble），保留分支以便扩展
            fused = _reciprocal_rank_fusion(
                [dense_docs, sparse_docs],
                [self.settings.rrf_weight_dense, self.settings.rrf_weight_sparse],
                c=self.settings.rrf_c,
            )
        elif retrieve_sparse:
            fused = sparse_docs
        else:
            fused = dense_docs

        return RetrievalResult(
            documents=fused[:limit],
            dense_documents=dense_docs,
            sparse_documents=sparse_docs,
            query=query,
        )

    async def aretrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        """异步检索（EnsembleRetriever 的异步路径）。"""
        self.ensure_ready()
        limit = top_k or self.settings.fusion_top_k
        fused = list(await self._ensemble.ainvoke(query))  # type: ignore[union-attr]
        return RetrievalResult(
            documents=fused[:limit],
            dense_documents=[],
            sparse_documents=[],
            query=query,
        )

    def retrieve_with_scores(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """融合结果 + 向量相似度分数。

        Agent 的回答质量依赖「相关性门控」，而 RRF 分数本身是无量纲的排名倒数和，
        不能直接当阈值用。这里用向量 COSINE 相似度作为绝对相关性度量，
        找不到对应向量分数的片段（例如仅被 BM25 召回的）记 0 分。
        """
        result = self.retrieve(query, top_k=top_k)
        return self.score_documents(query, result.documents, top_k=top_k)

    def score_documents(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """给一批已检索到的片段补上向量相似度分数。

        异步检索路径拿不到分数（``EnsembleRetriever`` 只返回文档），
        用本方法补分即可复用同一套门控逻辑，避免「同步路径会拒答、
        异步路径却把弱相关片段丢给大模型」的不一致。
        """
        if not documents:
            return []

        from app.rag.vectorstore import get_vector_store

        score_map: dict[str, float] = {}
        for doc, score in get_vector_store().similarity_search_with_score(
            query, k=max(self.settings.dense_top_k, top_k or self.settings.fusion_top_k)
        ):
            key = str(doc.metadata.get(RRF_ID_KEY) or doc.page_content)
            # COSINE 相似度在部分 Milvus 版本中以距离形式返回（越大越不相关），
            # 这里统一折算为「越大越相关」并夹到 [0, 1]
            normalized = float(score)
            if normalized > 1.0:
                normalized = 1.0 / (1.0 + normalized) if normalized > 2.0 else 1.0 - normalized
            score_map.setdefault(key, max(0.0, min(1.0, normalized)))

        scored: list[tuple[Document, float]] = []
        for doc in documents:
            key = str(doc.metadata.get(RRF_ID_KEY) or doc.page_content)
            scored.append((doc, score_map.get(key, 0.0)))
        return scored


def _reciprocal_rank_fusion(
    doc_lists: Sequence[Sequence[Document]],
    weights: Sequence[float],
    *,
    c: int = 60,
) -> list[Document]:
    """纯函数版加权 RRF，作为 ``EnsembleRetriever`` 不可用时的等价兜底。"""
    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}

    for docs, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            key = str(doc.metadata.get(RRF_ID_KEY) or doc.page_content)
            scores[key] = scores.get(key, 0.0) + weight / (rank + c)
            first_seen.setdefault(key, doc)

    return [first_seen[key] for key in sorted(scores, key=lambda k: scores[k], reverse=True)]


def format_documents(documents: Iterable[Document], *, max_chars: int = 1400) -> str:
    """把检索片段格式化为带编号的上下文块，供 Prompt 引用。"""
    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        section = metadata.get("section") or metadata.get("title") or "未知章节"
        source = metadata.get("source", "未知来源")
        content = doc.page_content.strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        blocks.append(f"[{index}] 来源：{source}｜章节：{section}\n{content}")
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def get_hybrid_retriever_cached() -> HybridPolicyRetriever:
    """获取全局唯一混合检索器实例（**不加载体，lazy**）。

    这里刻意**不**调用 ``reload()``。原因：``reload()`` 会载入 BM25 语料、
    连接 Milvus 并在首次查询时加载向量模型，是秒级甚至十几秒的重操作。
    若在此处执行，调用方（如 FastAPI lifespan）会在事件循环里同步阻塞。

    加载统一交给 ``reload()`` / ``ensure_ready()``：
    * 服务启动：``app.main`` 的 lifespan 用 ``asyncio.to_thread`` 调 ``reload()``；
    * 其它场景（脚本、测试）：首次 ``retrieve()`` 时由 ``ensure_ready()`` 懒加载。

    Returns:
        尚未加载语料的检索器实例；使用前 ``ensure_ready()`` 会保证就绪。
    """
    return HybridPolicyRetriever()


def get_hybrid_retriever() -> HybridPolicyRetriever:
    """获取全局混合检索器（语义化别名）。"""
    return get_hybrid_retriever_cached()
