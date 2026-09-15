"""Milvus 向量库封装。

固定使用 **显式 schema + 关闭 dynamic field**：政策片段的元数据字段是已知的，
显式建表可以让 Milvus 在过滤条件上走索引，也避免脏字段污染集合。

集合结构（默认名 ``tenhub_policy_chunks``）::

    chunk_id     VARCHAR(128)  主键   -> {"chunk_id": "..."}   业务去重键
    text         VARCHAR(8192) 文本   -> {"text": "..."}       片段正文
    vector       FLOAT_VECTOR  512    向量
    source/title/section/doc_type/chunk_index/file_hash/source_path
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from langchain_core.documents import Document

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 写入 Milvus 的元数据字段（顺序即 schema 顺序）
METADATA_FIELDS: tuple[str, ...] = (
    "source",
    "source_path",
    "file_hash",
    "doc_type",
    "title",
    "section",
    "chunk_index",
)

# 主键字段名与真正承载文本的字段名（LangChain 约定用 text_field 存正文）
PRIMARY_FIELD = "chunk_id"
TEXT_FIELD = "text"
VECTOR_FIELD = "vector"

# Milvus VARCHAR 上限，超长片段会被拒绝写入
_MAX_TEXT_LEN = 8192

def build_milvus_schema(client: Any, dim: int) -> Any:
    """构造显式 Milvus schema。"""
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field(PRIMARY_FIELD, DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=_MAX_TEXT_LEN)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("source_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("file_hash", DataType.VARCHAR, max_length=128)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=16)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("section", DataType.VARCHAR, max_length=1024)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    # 归一化向量 + COSINE，等价于内积；HNSW 在中小规模下召回与延迟均衡
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    return schema, index_params


class PolicyVectorStore:
    """政策知识库的 Milvus 访问层。"""

    def __init__(self, settings: Settings | None = None, *, lazy: bool = True) -> None:
        self.settings = settings or default_settings
        self._store: Any = None
        self._lazy = lazy

    # ------------------------------------------------------------------
    # 连接与集合
    # ------------------------------------------------------------------
    def _build_store(self) -> Any:
        from langchain_milvus import Milvus

        from app.rag.embeddings import get_embeddings

        embeddings = get_embeddings(self.settings)
        connection_args: dict[str, Any] = {
            "uri": self.settings.milvus_uri,
            "timeout": self.settings.milvus_timeout,
        }
        if self.settings.milvus_token:
            connection_args["token"] = self.settings.milvus_token

        return Milvus(
            embedding_function=embeddings,
            collection_name=self.settings.milvus_collection,
            connection_args=connection_args,
            primary_field=PRIMARY_FIELD,
            text_field=TEXT_FIELD,
            vector_field=VECTOR_FIELD,
            # 元数据在 schema 中是**顶层标量字段**（见 build_milvus_schema），
            # 因此必须显式关闭 JSON 元数据聚合，否则 LangChain 会尝试写入
            # 一个不存在的 "metadata" 字段而报错。
            metadata_field=None,
            enable_dynamic_field=False,
            auto_id=False,
            consistency_level="Strong",
        )

    @property
    def store(self) -> Any:
        """LangChain Milvus VectorStore 实例（惰性构建）。"""
        if self._store is None:
            self._store = self._build_store()
        return self._store

    def reset(self) -> None:
        """删除并重建集合（全量重建索引时使用）。"""
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=self.settings.milvus_uri,
            token=self.settings.milvus_token or None,
            timeout=self.settings.milvus_timeout,
        )
        name = self.settings.milvus_collection
        if client.has_collection(name):
            logger.warning("删除已有 Milvus 集合: %s", name)
            client.drop_collection(name)
        self._store = None

    def collection_exists(self) -> bool:
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=self.settings.milvus_uri,
            token=self.settings.milvus_token or None,
            timeout=self.settings.milvus_timeout,
        )
        return bool(client.has_collection(self.settings.milvus_collection))

    def count(self) -> int:
        """统计集合内实体数量。

        优先用 ``query(count(*))``（实时准确）；``get_collection_stats`` 依赖
        flush 后的统计信息，刚写入时可能仍返回 0，故仅作兜底。
        """
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(
                uri=self.settings.milvus_uri,
                token=self.settings.milvus_token or None,
                timeout=self.settings.milvus_timeout,
            )
            name = self.settings.milvus_collection
            if not client.has_collection(name):
                return 0
            try:
                result = client.query(
                    collection_name=name,
                    filter="",
                    output_fields=["count(*)"],
                )
                if result:
                    return int(result[0].get("count(*)", 0))
            except Exception as exc:  # noqa: BLE001 - 退化为统计信息
                logger.debug("count(*) 查询失败，回退 collection_stats: %s", exc)
            stats = client.get_collection_stats(name)
            return int(stats.get("row_count", 0))
        except Exception as exc:  # noqa: BLE001 - 统计失败不影响主流程
            logger.warning("统计 Milvus 行数失败: %s", exc)
            return 0

    def flush(self) -> None:
        """把内存中的数据刷入持久化段。

        仅影响 ``get_collection_stats`` 等基于段统计的接口；检索本身无需 flush。
        """
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(
                uri=self.settings.milvus_uri,
                token=self.settings.milvus_token or None,
                timeout=self.settings.milvus_timeout,
            )
            if client.has_collection(self.settings.milvus_collection):
                client.flush(self.settings.milvus_collection)
        except Exception as exc:  # noqa: BLE001
            logger.debug("flush Milvus 失败: %s", exc)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def recreate_collection(self) -> None:
        """按当前 embedding 维度重建集合。"""
        from pymilvus import MilvusClient

        from app.rag.embeddings import get_embeddings

        dim = get_embeddings(self.settings).dimension
        client = MilvusClient(
            uri=self.settings.milvus_uri,
            token=self.settings.milvus_token or None,
            timeout=self.settings.milvus_timeout,
        )
        name = self.settings.milvus_collection
        if client.has_collection(name):
            logger.warning("重建集合，删除旧的 %s", name)
            client.drop_collection(name)

        schema, index_params = build_milvus_schema(client, dim)
        client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        logger.info("已创建 Milvus 集合 %s (dim=%d, metric=COSINE)", name, dim)
        self._store = None

    def _prepare_documents(self, documents: Sequence[Document]) -> list[Document]:
        """清洗后返回可写入的文档（截断超长文本、补齐主键）。"""
        prepared: list[Document] = []
        for index, doc in enumerate(documents):
            text = doc.page_content or ""
            if not text.strip():
                continue
            if len(text) > _MAX_TEXT_LEN:
                logger.warning(
                    "片段超长已截断 (%d -> %d): %s",
                    len(text),
                    _MAX_TEXT_LEN,
                    doc.metadata.get("chunk_id", index),
                )
                text = text[:_MAX_TEXT_LEN]

            metadata = {k: v for k, v in doc.metadata.items() if k in METADATA_FIELDS}
            # Milvus 标量字段不接受 None
            metadata["chunk_index"] = int(metadata.get("chunk_index") or 0)
            for key in METADATA_FIELDS:
                if key == "chunk_index":
                    continue
                value = metadata.get(key)
                metadata[key] = "" if value is None else str(value)[:1024]

            prepared.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                    id=doc.metadata.get("chunk_id") or doc.id or f"auto-{index:08d}",
                )
            )
        return prepared

    def add_documents(self, documents: Sequence[Document], *, batch_size: int = 64) -> int:
        """批量写入向量库，返回成功写入的片段数。"""
        prepared = self._prepare_documents(documents)
        if not prepared:
            return 0

        store = self.store
        written = 0
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            ids = [doc.id for doc in batch]
            store.add_documents(documents=batch, ids=ids)
            written += len(batch)
            logger.info("已写入 Milvus %d/%d", written, len(prepared))

        # 写入后 flush，使段统计（get_collection_stats 等）立即可用
        self.flush()
        return written

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def as_retriever(self, *, k: int | None = None) -> Any:
        """返回相似度检索器（供 EnsembleRetriever 使用）。"""
        top_k = k or self.settings.dense_top_k
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """带相似度分数的检索，分数为 COSINE 相似度（越大越相关）。"""
        top_k = k or self.settings.dense_top_k
        try:
            return self.store.similarity_search_with_score(query, k=top_k)
        except Exception as exc:  # noqa: BLE001 - 集合为空等情况降级为无结果
            logger.error("Milvus 检索失败: %s", exc)
            return []


@lru_cache(maxsize=1)
def get_vector_store() -> PolicyVectorStore:
    """全局唯一向量库访问层。"""
    return PolicyVectorStore()
