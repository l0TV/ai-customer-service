"""请求上下文：把 JWT 经 ``RunnableConfig`` 透传给工具。

LangChain 的 ``RunnableConfig`` 会贯穿整条调用链（Agent -> 工具），
因此它是承载「每次请求的凭据」的天然通道。

* 存放位置：``config["configurable"][JWT_CONFIG_KEY]``
* 写入方：FastAPI 路由层（本服务唯一的凭据入口）
* 读取方：工具函数（见 ``app/tools/complaint_tools.py``）

工具形参用 ``Annotated[..., InjectedToolArg]`` 标注，LangChain 不会把该
参数暴露给大模型，从而保证 JWT 永远不出现在模型可见的 tool schema 里，
模型也无法伪造或篡改它。
"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.runnables import RunnableConfig

# RunnableConfig 中承载 JWT 的键名
JWT_CONFIG_KEY = "jwt_token"
# 承载会话/追踪标识的键名
SESSION_CONFIG_KEY = "session_id"
USER_TAG_CONFIG_KEY = "user_tag"

_MISSING = object()


class MissingJwtContextError(RuntimeError):
    """工具执行时未能在 RunnableConfig 中找到 JWT。"""


def build_run_config(
    jwt_token: str,
    *,
    session_id: str | None = None,
    user_tag: str | None = None,
) -> RunnableConfig:
    """构造带有 JWT 的 ``RunnableConfig``。

    Args:
        jwt_token: 原始用户 JWT（不含 ``Bearer `` 前缀）。
        session_id: 可选会话标识，用于日志串联与多轮记忆。
        user_tag: 可选的脱敏用户标识（仅用于日志）。

    Returns:
        可直接传给 ``agent.ainvoke(..., config=...)`` 的配置字典。
    """
    configurable: dict[str, Any] = {JWT_CONFIG_KEY: jwt_token}
    if session_id:
        configurable[SESSION_CONFIG_KEY] = session_id
    if user_tag:
        configurable[USER_TAG_CONFIG_KEY] = user_tag

    return {"configurable": configurable}


def _lookup(config: Mapping[str, Any] | None, key: str) -> Any:
    """在 RunnableConfig 及其嵌套结构中查找键。

    LangGraph 在不同位置传递 config（``configurable`` 或顶层），
    这里做一次宽松查找，避免因版本差异导致透传失效。
    """
    if not config:
        return _MISSING

    if key in config:
        return config[key]

    configurable = config.get("configurable")
    if isinstance(configurable, Mapping) and key in configurable:
        return configurable[key]

    metadata = config.get("metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]

    return _MISSING


def get_jwt_from_config(config: RunnableConfig | Mapping[str, Any] | None) -> str:
    """从 ``RunnableConfig`` 中取出 JWT。

    Raises:
        MissingJwtContextError: 配置中不存在 JWT，说明调用方漏传，
            此时必须失败而不是匿名调用后端，避免越权。
    """
    value = _lookup(config, JWT_CONFIG_KEY)
    if value is _MISSING or not isinstance(value, str) or not value.strip():
        raise MissingJwtContextError(
            "RunnableConfig 中缺少 jwt_token，无法携带用户身份调用后端接口。"
            "请确认路由层使用 build_run_config() 构造了 config。"
        )
    return value.strip()


def get_jwt_from_tool_runtime(runtime: Any) -> str:
    """从 LangGraph 注入的 ``ToolRuntime`` 中取出 JWT。

    这是工具获取凭据的**推荐入口**。原因：``InjectedToolArg`` 标注的 config 形参
    在当前 LangChain 版本中并不会被自动注入（实测传入的默认值为 None），
    而 ``ToolRuntime.config`` 由 LangGraph 的工具执行节点保证填充，
    且该形参同样不会出现在模型可见的 tool schema 中。

    Args:
        runtime: 形参名为 ``runtime``、类型标注为 ``ToolRuntime`` 时由框架注入的对象。

    Raises:
        MissingJwtContextError: runtime 缺失或其中没有 JWT。
    """
    if runtime is None:
        raise MissingJwtContextError(
            "工具未收到 runtime 参数，无法获取用户身份。"
            "请确认工具签名的第一个（或带默认值的）参数名为 runtime 且类型标注为 ToolRuntime。"
        )

    config = getattr(runtime, "config", None)
    return get_jwt_from_config(config)


def get_session_id_from_tool_runtime(runtime: Any) -> str | None:
    """从 ``ToolRuntime`` 中取出会话标识（仅用于日志）。"""
    if runtime is None:
        return None
    return get_session_id_from_config(getattr(runtime, "config", None))


def token_from_runtime_or_args(runtime: Any, *args: Any) -> str:
    """尽力从多个位置取 JWT：ToolRuntime 优先，其次扫描入参中疑似 config 的字典。

    这样即使调用方（例如测试代码或自定义执行器）没有走 LangGraph 的 runtime 注入，
    只要把 ``{"configurable": {"jwt_token": ...}}`` 混在入参里，也能正常取到凭据，
    从而避免因执行链路差异导致工单静默失败。
    """
    if runtime is not None:
        try:
            return get_jwt_from_tool_runtime(runtime)
        except MissingJwtContextError:
            pass

    for arg in args:
        if isinstance(arg, Mapping) and (
            JWT_CONFIG_KEY in arg
            or (isinstance(arg.get("configurable"), Mapping) and JWT_CONFIG_KEY in arg["configurable"])
        ):
            return get_jwt_from_config(arg)

    raise MissingJwtContextError(
        "未能在 ToolRuntime 或工具入参中找到 jwt_token，无法携带用户身份调用后端接口。"
    )


def get_session_id_from_config(config: RunnableConfig | Mapping[str, Any] | None) -> str | None:
    value = _lookup(config, SESSION_CONFIG_KEY)
    return value if isinstance(value, str) and value.strip() else None


def get_user_tag_from_config(config: RunnableConfig | Mapping[str, Any] | None) -> str | None:
    value = _lookup(config, USER_TAG_CONFIG_KEY)
    return value if isinstance(value, str) and value.strip() else None
