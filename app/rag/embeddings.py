"""向量化（Embedding）封装。

使用本地 sentence-transformers 模型（默认 ``BAAI/bge-small-zh-v1.5``，512 维），
完全离线，不依赖任何外部 API。

关键细节：BGE 系列中文模型在 **查询侧** 需要加官方指令前缀
（``为这个句子生成表示以用于检索相关文章：``），文档侧则不加。
漏掉前缀会明显拉低语义检索召回率，因此这里显式区分
``embed_query`` / ``embed_documents``。
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 需要加查询指令前缀的模型家族（中文/多语言 BGE）
_INSTRUCTION_MODELS = ("bge-small-zh", "bge-base-zh", "bge-large-zh", "bge-m3")

# 权重文件名，用于判断模型是否已完整缓存
_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _is_local_dir(model_name: str) -> bool:
    try:
        return Path(model_name).exists()
    except OSError:
        return False


def _is_model_cached(model_name: str) -> bool:
    """判断模型是否已存在于 HF 本地缓存（config + 任一权重文件命中）。

    该判定不导入 transformers，可安全地在模块加载早期调用。
    判定失败一律按「未缓存」处理，不影响正常下载。
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False

    try:
        if not isinstance(try_to_load_from_cache(model_name, "config.json"), str):
            return False
        return any(
            isinstance(try_to_load_from_cache(model_name, name), str)
            for name in _WEIGHT_FILES
        )
    except Exception:  # noqa: BLE001
        return False


def _configure_offline_mode(model_name: str, force_offline: bool) -> bool:
    """在任何 HF 相关库被导入之前开启离线模式。

    **为什么必须在导入期设置**：``huggingface_hub`` 与 ``transformers`` 在
    import 时就把 ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` 读取为模块级
    常量，之后再 ``os.environ[...] = ...`` **不会生效**。而判断「模型是否已缓存」
    本身又需要 import huggingface_hub，会把这个顺序彻底搞乱——所以这里采用
    「由配置显式决定」的方式，不依赖缓存探测。

    不设为离线的后果（实测）：加载模型时会额外联网探测
    ``adapter_config.json`` / ``processor_config.json`` 等可选文件
    （普通非 LoRA 模型并不需要），网络不通时触发 5 次超时重试，
    把首次加载从约 1 秒拖到 14 秒，极端情况下直接把整个请求挂死。

    Args:
        model_name: 模型标识（仓库 ID 或本地目录）。
        force_offline: 配置开关 ``AI_CS_HF_OFFLINE``。

    Returns:
        是否启用离线模式。
    """
    if _is_local_dir(model_name):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return True

    if force_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return True

    # 允许联网：清掉可能由外部传入的离线开关，确保首次能下载
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    return False


# 模块导入即执行：此刻尚未导入任何 HF 相关库，设置才真正有效
_OFFLINE_MODE = _configure_offline_mode(
    default_settings.embedding_model, default_settings.hf_offline
)


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

            # 若本次加载的模型与模块导入时判定的那个不同（例如运行时改了配置），
            # 再兜底判定一次。注意：此时 HF 库可能已导入，设置环境变量未必生效，
            # 因此正常路径依赖模块顶部的 _configure_offline_mode。
            if not _OFFLINE_MODE and (
                _is_local_dir(self.model_name) or _is_model_cached(self.model_name)
            ):
                logger.warning(
                    "模型 %s 已缓存，但离线模式未在导入期启用；"
                    "若加载卡顿/超时，请设置 HF_HUB_OFFLINE=1 后重启服务。",
                    self.model_name,
                )

            from sentence_transformers import SentenceTransformer

            model_ref = (
                str(Path(self.model_name).resolve())
                if _is_local_dir(self.model_name)
                else self.model_name
            )

            logger.info(
                "正在加载向量模型 %s (device=%s, offline=%s)…",
                model_ref,
                self._device,
                _OFFLINE_MODE,
            )
            model = SentenceTransformer(model_ref, device=self._device)
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
