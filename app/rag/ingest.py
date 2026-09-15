"""知识库入库编排。

流程::

    扫描文档 -> 解析(PDF/Word/MD/TXT) -> 按标题分章节 -> 递归分块
        -> 写 Milvus(向量) -> 落盘 BM25 语料快照 -> 重建混合检索器

关键点：**BM25 语料与 Milvus 必须同源同步**。两者都来自同一批分块结果，
只要在写 Milvus 成功后立刻刷新快照，就不会出现「关键词能搜到、向量搜不到」
的不一致状态。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging_config import get_logger
from app.rag.hybrid_retriever import get_hybrid_retriever_cached, save_bm25_corpus
from app.rag.loader import chunk_document, deduplicate, iter_policy_files, load_document

logger = get_logger(__name__)


def _reload_retriever() -> None:
    """入库后刷新混合检索器，使新片段立即可检索。"""
    retriever = get_hybrid_retriever_cached()
    rebuilt = retriever.reload()
    logger.info(
        "混合检索器已刷新：BM25 语料 %d 条，混合模式=%s",
        retriever.corpus_size,
        "BM25+向量" if rebuilt else "纯向量",
    )


def _ingest_sync(
    files: list[Path],
    *,
    rebuild: bool,
) -> dict[str, Any]:
    """同步执行入库主体逻辑（在线程池中调用）。"""
    from app.rag.vectorstore import get_vector_store

    started = time.perf_counter()
    store = get_vector_store()

    # --- 1. 全量重建集合 ---
    if rebuild:
        store.recreate_collection()

    # --- 2. 解析与分块 ---
    all_chunks: list[Document] = []
    details: list[dict[str, Any]] = []

    for path in files:
        try:
            loaded = load_document(path)
        except Exception as exc:  # noqa: BLE001 - 单文件失败不阻断整批
            logger.error("解析失败 %s: %s", path.name, exc)
            details.append({"file": path.name, "ok": False, "error": str(exc)})
            continue

        chunks = chunk_document(loaded, settings=settings)
        all_chunks.extend(chunks)
        details.append(
            {
                "file": path.name,
                "ok": True,
                "chars": len(loaded.text),
                "chunks": len(chunks),
            }
        )

    all_chunks = deduplicate(all_chunks)
    logger.info("解析完成：%d 个文件 -> %d 个片段", len(files), len(all_chunks))

    # --- 3. 写入 Milvus ---
    written = 0
    if all_chunks:
        written = store.add_documents(all_chunks)
    else:
        logger.warning("没有可写入的片段")

    # --- 4. 刷新 BM25 语料快照 ---
    # 注意：全量重建时必须整体覆盖快照；增量追加时把新旧语料合并去重。
    if rebuild or not settings.bm25_corpus_path.exists():
        save_bm25_corpus(all_chunks, settings.bm25_corpus_path)
    elif all_chunks:
        from app.rag.hybrid_retriever import load_bm25_corpus

        merged = deduplicate([*load_bm25_corpus(settings.bm25_corpus_path), *all_chunks])
        save_bm25_corpus(merged, settings.bm25_corpus_path)

    # --- 5. 刷新检索器 ---
    _reload_retriever()

    elapsed = time.perf_counter() - started
    return {
        "files": len([d for d in details if d.get("ok")]),
        "documents": len(files),
        "chunks": len(all_chunks),
        "written": written,
        "collection_count": store.count(),
        "elapsed_seconds": round(elapsed, 2),
        "details": details,
    }


async def ingest_policies(
    *,
    rebuild: bool = False,
    directory: str | None = None,
) -> dict[str, Any]:
    """把政策文档目录全量（或增量）入库。

    Args:
        rebuild: 是否先清空并重建 Milvus 集合。
        directory: 自定义文档目录，默认 ``data/policies``。

    Returns:
        统计字典，字段与 ``IngestResponse`` 对应。
    """
    target = Path(directory) if directory else settings.policy_dir
    if not target.exists():
        raise FileNotFoundError(f"政策文档目录不存在: {target}")

    files = list(iter_policy_files(target, settings.supported_suffixes))
    if not files:
        logger.warning("目录 %s 下没有找到受支持的文档", target)
        return {
            "files": 0,
            "documents": 0,
            "chunks": 0,
            "written": 0,
            "collection_count": 0,
            "elapsed_seconds": 0.0,
            "details": [],
        }

    logger.info("开始入库：目录=%s 文件数=%d rebuild=%s", target, len(files), rebuild)
    # 解析 PDF/Word 与向量化都是 CPU 密集的阻塞操作，放线程池避免卡住事件循环
    return await asyncio.to_thread(_ingest_sync, files, rebuild=rebuild)


async def ingest_uploaded_file(
    filename: str,
    content: bytes,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    """保存上传的文件到政策目录后入库。"""
    # 只取文件名，防止路径穿越
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("无效的文件名")

    target = settings.policy_dir / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    logger.info("已保存上传文件: %s (%d 字节)", target, len(content))

    return await ingest_policies(rebuild=rebuild, directory=str(settings.policy_dir))
