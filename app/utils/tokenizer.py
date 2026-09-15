"""中文分词工具。

BM25 依赖词元（token）粒度统计。LangChain 的 ``BM25Retriever`` 默认使用
英文空白分词（``default_preprocessing_func``），对中文会把整句当成一个 token，
召回效果接近于零。这里用 jieba 做中文分词，并过滤空白与纯标点词元。

同一个分词函数必须同时用于「建索引」与「查询」，否则 BM25 统计口径不一致。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

# 纯标点/空白/控制字符，参与 BM25 统计会引入噪声
_PUNCT_ONLY = re.compile(r"^[\s\W_]+$", re.UNICODE)

# 常见中文停用词，去掉后可显著提升政策类文本的关键词区分度
_STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 和 是 就 都 而 及 与 或 在 有 为 对 到 以 之 其 于 上 下 中 里
    这 那 这个 那个 这些 那些 一个 一种 我们 你们 他们 它们 自己 什么 怎么
    如何 可以 应该 需要 请 问 吗 呢 吧 啊 哦 嗯 会 能 要 被 把 让 给 从 向
    并且 但是 因为 所以 如果 那么 不过 然后 还是 也是 不是 没有 一些 该
    """.split()
)


@lru_cache(maxsize=4)
def _tokenizer(mode: str):
    """惰性导入 jieba，避免服务启动即加载词典。"""
    import jieba

    jieba.setLogLevel(60)  # 关闭 jieba 的 "Building prefix dict" 日志
    if mode == "search":
        # 搜索引擎模式：对长词再切分，提高召回率
        return lambda text: jieba.lcut_for_search(text)
    return lambda text: jieba.lcut(text)


def tokenize(text: str, *, mode: str = "search") -> list[str]:
    """把文本切成 BM25 词元列表。

    Args:
        text: 待分词文本。
        mode: ``search`` 用 jieba 搜索引擎模式（召回优先），
            ``default`` 用精确模式。

    Returns:
        过滤后的词元列表，已去除停用词与纯标点。
    """
    if not text or not text.strip():
        return []

    tokens: Iterable[str] = _tokenizer(mode)(text)

    result: list[str] = []
    for token in tokens:
        token = token.strip().lower()
        if not token or _PUNCT_ONLY.match(token) or token in _STOPWORDS:
            continue
        result.append(token)
    return result


def bm25_preprocess(text: str) -> list[str]:
    """供 ``BM25Retriever(preprocess_func=...)`` 使用的分词入口。"""
    return tokenize(text, mode="search")
