"""向量化（Embedding）封装。

使用本地 sentence-transformers 模型（默认 ``BAAI/bge-small-zh-v1.5``，512 维），
完全离线，不依赖任何外部 API。

关键细节：BGE 系列中文模型在 **查询侧** 需要加官方指令前缀
（``为这个句子生成表示以用于检索相关文章：``），文档侧则不加。
漏掉前缀会明显拉低语义检索召回率，因此这里显式区分
``embed_query`` / ``embed_documents``。
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 需要加查询指令前缀的模型家族（中文/多语言 BGE）
_INSTRUCTION_MODELS = ("bge-small-zh", "bge-base-zh", "bge-large-zh", "bge-m3")


class BGEEmbeddings(Embeddings):
    """带查询指令前缀的本地 BGE 向量化器。

    包一层 ``HuggingFaceEmbeddings`` 而不是直接使用，是因为需要
    对 query 加前缀、对 document 不加，而官方封装不做这个区分。
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        normalize: bool = True,
        query_instruction: str = "",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size

        # 延迟到首次使用再加载，缩短服务冷启动时间
        self._lock = threading.Lock()
        self._model: Any = None
        self._device = device
        self._normalize = normalize

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            from sentence_transformers import SentenceTransformer

            logger.info("正在加载向量模型 %s (device=%s)…", self.model_name, self._device)
            model = SentenceTransformer(self.model_name, device=self._device)
            self._model = model
            dim = model.get_embedding_dimension()
            logger.info("向量模型加载完成，维度=%s", dim)
            return self._model

    @property
    def dimension(self) -> int:
        return int(self._ensure_model().get_embedding_dimension())

    # ------------------------------------------------------------------
    # Embeddings 接口
    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档侧向量化：不加指令前缀。"""
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """查询侧向量化：按需加 BGE 指令前缀。"""
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")
        prepared = f"{self.query_instruction}{text}" if self.query_instruction else text
        model = self._ensure_model()
        vector = model.encode(
            [prepared],
            batch_size=1,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # CPU 推理本身受 GIL 限制，异步包装无收益；直接复用同步实现
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _needs_instruction(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(family in lowered for family in _INSTRUCTION_MODELS)


@lru_cache(maxsize=4)
def _build_embeddings(
    model_name: str,
    device: str,
    normalize: bool,
    query_instruction: str,
) -> BGEEmbeddings:
    return BGEEmbeddings(
        model_name,
        device=device,
        normalize=normalize,
        query_instruction=query_instruction,
    )


def get_embeddings(settings: Settings | None = None) -> BGEEmbeddings:
    """获取全局向量化器（按配置缓存）。"""
    cfg = settings or default_settings
    instruction = (
        cfg.embedding_query_instruction if _needs_instruction(cfg.embedding_model) else ""
    )
    if not instruction and _needs_instruction(cfg.embedding_model):
        logger.warning("向量模型 %s 建议配置查询指令前缀", cfg.embedding_model)
    return _build_embeddings(
        cfg.embedding_model,
        cfg.embedding_device,
        cfg.embedding_normalize,
        instruction,
    )


def embeddings_available(settings: Settings | None = None) -> bool:
    """探测向量模型是否可加载（不抛异常）。"""
    try:
        get_embeddings(settings)._ensure_model()
        return True
    except Exception as exc:  # noqa: BLE001 - 健康检查不应抛出
        logger.error("向量模型不可用: %s", exc)
        return False
