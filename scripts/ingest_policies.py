"""知识库入库 CLI。

用途与 HTTP 接口 ``POST /api/ai/rag/ingest`` 等价，适合初始化与运维脚本化执行。

用法::

    # 首次入库：解析 data/policies 下全部文档，重建 Milvus 集合并刷新 BM25 快照
    python scripts/ingest_policies.py --rebuild

    # 增量：新增文档后只追加（不删除已有数据）
    python scripts/ingest_policies.py

    # 指定其它目录
    python scripts/ingest_policies.py --rebuild --dir D:\\policies

    # 只看会处理哪些文件，不真正写入
    python scripts/ingest_policies.py --dry-run
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
from app.rag.hybrid_retriever import load_bm25_corpus  # noqa: E402
from app.rag.ingest import ingest_policies  # noqa: E402
from app.rag.loader import iter_policy_files  # noqa: E402

import asyncio  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="政策文档入库到 Milvus + BM25")
    parser.add_argument("--rebuild", action="store_true", help="清空并重建 Milvus 集合", default=True)
    parser.add_argument("--dir", default=None, help="文档目录，默认 data/policies")
    parser.add_argument("--dry-run", action="store_true", help="只列出待处理文件")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    args = parser.parse_args()

    setup_logging(args.log_level)

    target = Path(args.dir) if args.dir else settings.policy_dir
    files = list(iter_policy_files(target, settings.supported_suffixes))

    print(f"文档目录 : {target}")
    print(f"Milvus   : {settings.milvus_uri} / {settings.milvus_collection}")
    print(f"向量模型 : {settings.embedding_model} ({settings.embedding_dim} 维)")
    print(f"分块参数 : size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    print(f"待处理   : {len(files)} 个文件")
    for path in files:
        try:
            size_kb = path.stat().st_size / 1024
            print(f"  - {path.name} ({size_kb:.1f} KB)")
        except OSError:
            print(f"  - {path.name}")

    if not files:
        print("\n没有找到受支持的文档（.md/.markdown/.txt/.pdf/.docx），请先放入政策文件。")
        return 1

    if args.dry_run:
        print("\n--dry-run：未执行写入。")
        return 0

    print("\n开始入库…（首次会加载向量模型并下载，请耐心等待）")
    stats = asyncio.run(ingest_policies(rebuild=args.rebuild, directory=str(target)))

    print("\n" + "=" * 60)
    print("入库完成")
    print(f"  成功文件   : {stats['files']} / {stats['documents']}")
    print(f"  生成片段   : {stats['chunks']}")
    print(f"  写入 Milvus: {stats['written']}")
    print(f"  集合总量   : {stats['collection_count']}")
    print(f"  耗时       : {stats['elapsed_seconds']} 秒")
    print(f"  BM25 语料  : {settings.bm25_corpus_path}")

    failures = [item for item in stats["details"] if not item.get("ok")]
    if failures:
        print(f"\n失败文件（{len(failures)} 个）:")
        for item in failures:
            print(f"  - {item['file']}: {item.get('error')}")
    else:
        print("  全部文件解析成功")

    corpus = load_bm25_corpus(settings.bm25_corpus_path)
    print(f"  BM25 快照条数: {len(corpus)}")
    print("=" * 60)

    if stats["chunks"] != len(corpus):
        print("\n⚠ 警告：Milvus 写入片段数与 BM25 快照条数不一致，")
        print("  可能导致关键词检索与向量检索结果不同源。建议加 --rebuild 重跑。")
        return 2

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
