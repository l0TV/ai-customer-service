"""大模型客户端构造。

默认对接 DeepSeek（OpenAI 兼容协议）。把构造逻辑集中在这里，
便于替换为任意 OpenAI 兼容端点（含本地 vLLM / Ollama 的 /v1 接口）。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def build_chat_model(
    settings: Settings | None = None,
    *,
    temperature: float | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """构造对话模型。

    Args:
        settings: 配置对象，默认取全局配置。
        temperature: 覆盖默认温度。
        streaming: 是否启用流式输出。

    Raises:
        ValueError: 未配置 API Key。
    """
    cfg = settings or default_settings

    api_key = cfg.deepseek_api_key.strip()
    if not api_key:
        raise ValueError(
            "未配置 AI_CS_DEEPSEEK_API_KEY，无法调用大模型。"
            "请在 ai-customer-service/.env 中设置该变量。"
        )

    logger.info("初始化对话模型: %s @ %s", cfg.chat_model, cfg.deepseek_base_url)
    return ChatOpenAI(
        model=cfg.chat_model,
        api_key=api_key,
        base_url=cfg.deepseek_base_url,
        temperature=cfg.temperature if temperature is None else temperature,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        streaming=streaming,
    )


@lru_cache(maxsize=4)
def get_chat_model(temperature: float | None = None) -> BaseChatModel:
    """获取全局对话模型实例。"""
    return build_chat_model(temperature=temperature)
