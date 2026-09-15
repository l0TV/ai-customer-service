"""统一日志配置。"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """初始化根日志器，重复调用安全。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # 降低三方库噪声，保留告警以上
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "jieba",
        "sentence_transformers",
        "transformers",
        "pymilvus",
        "milvus_lite",
        "openai",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
