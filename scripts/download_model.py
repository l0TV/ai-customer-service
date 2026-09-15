"""向量模型下载脚本。

默认向量模型 ``BAAI/bge-small-zh-v1.5`` 需要在首次运行时下载（约 95MB）。
国内直连 HuggingFace 经常超时，本脚本提供两条镜像路径：

1. **ModelScope（魔搭）**：国内速度最快，BAAI 官方模型在此有同步仓库 —— 默认优先；
2. **HF 镜像**：``hf-mirror.com``，通过 ``HF_ENDPOINT`` 生效。

用法::

    python scripts/download_model.py
    python scripts/download_model.py --model BAAI/bge-m3
    python scripts/download_model.py --source hf-mirror
    python scripts/download_model.py --source hf   # 直连，需可访问 huggingface.co

下载完成后会把模型落到 HF 缓存目录，之后 sentence-transformers 直接命中缓存，
离线也能正常加载。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许以脚本方式直接运行（把项目根加入 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
HF_MIRROR = "https://hf-mirror.com"


def _log(message: str) -> None:
    print(f"[download_model] {message}", flush=True)


def try_modelscope(model_id: str) -> str | None:
    """从 ModelScope 下载模型，返回本地目录。"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        _log("未安装 modelscope，跳过该路径（可 pip install modelscope）")
        return None

    _log(f"尝试从 ModelScope 下载 {model_id} …")
    try:
        local_dir = snapshot_download(model_id, cache_dir=str(Path.home() / ".cache"))
        _log(f"ModelScope 下载成功: {local_dir}")
        return str(local_dir)
    except Exception as exc:  # noqa: BLE001 - 失败即回退下一条路径
        _log(f"ModelScope 下载失败: {type(exc).__name__}: {exc}")
        return None


def try_sentence_transformers(model_id: str) -> str | None:
    """用 sentence-transformers 直接下载（遵循 HF_ENDPOINT 环境变量）。"""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    _log(
        f"使用 sentence-transformers 下载 {model_id}"
        + (f"（HF_ENDPOINT={endpoint}）" if endpoint else "（直连 huggingface.co）")
    )
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        dim = model.get_sentence_embedding_dimension()
        _log(f"下载并加载成功，向量维度 = {dim}")
        return model_id
    except Exception as exc:  # noqa: BLE001
        _log(f"sentence-transformers 下载失败: {type(exc).__name__}: {exc}")
        return None


def verify(model_id_or_path: str, expected_dim: int | None = None) -> bool:
    """加载模型并做一次真实编码，确认可用。"""
    try:
        from sentence_transformers import SentenceTransformer

        _log(f"验证模型可用性: {model_id_or_path} …")
        model = SentenceTransformer(model_id_or_path)
        dim = model.get_sentence_embedding_dimension()
        vector = model.encode(["退货政策是什么"], normalize_embeddings=True)
        _log(f"验证通过：维度={dim}，编码输出长度={len(vector[0])}")

        if expected_dim and dim != expected_dim:
            _log(
                f"⚠ 维度({dim}) 与配置 AI_CS_EMBEDDING_DIM({expected_dim}) 不一致！"
                "请同步修改配置，否则 Milvus 集合维度会不匹配。"
            )
        return True
    except Exception as exc:  # noqa: BLE001
        _log(f"验证失败: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并验证本地向量模型")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型 ID")
    parser.add_argument(
        "--source",
        choices=["auto", "modelscope", "hf-mirror", "hf"],
        default="auto",
        help="下载来源；auto 依次尝试 modelscope -> hf-mirror -> 直连",
    )
    parser.add_argument("--dim", type=int, default=512, help="期望向量维度，用于校验")
    parser.add_argument("--skip-verify", action="store_true", help="跳过验证")
    args = parser.parse_args()

    _log(f"目标模型: {args.model}")

    # 先看本地是否已有缓存
    if verify(args.model, args.dim) and not args.skip_verify:
        _log("模型已存在且可用，无需下载。")
        return 0

    sources = {
        "auto": ["modelscope", "hf-mirror", "hf"],
        "modelscope": ["modelscope"],
        "hf-mirror": ["hf-mirror"],
        "hf": ["hf"],
    }[args.source]

    for source in sources:
        if source == "hf-mirror":
            os.environ["HF_ENDPOINT"] = HF_MIRROR
            _log(f"已设置 HF_ENDPOINT={HF_MIRROR}")
        elif source == "hf":
            os.environ.pop("HF_ENDPOINT", None)

        if source == "modelscope":
            result = try_modelscope(args.model)
            if result and not args.skip_verify and verify(result, args.dim):
                _log("完成。")
                return 0
            if result:
                return 0
        else:
            result = try_sentence_transformers(args.model)
            if result and not args.skip_verify and verify(result, args.dim):
                _log("完成。")
                return 0
            if result:
                return 0

    _log("=" * 60)
    _log("所有下载路径均失败。可选处理方式：")
    _log("  1) 安装 modelscope 后重试： pip install modelscope")
    _log("  2) 手动下载模型目录，然后用本地路径启动：")
    _log("     AI_CS_EMBEDDING_MODEL=D:\\models\\bge-small-zh-v1.5")
    _log("  3) 配置代理后重试，例如： $env:HTTPS_PROXY='http://127.0.0.1:7890'")
    _log("=" * 60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
