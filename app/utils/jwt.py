"""JWT 透传工具。

设计约束（务必遵守，否则破坏安全边界）：

* 本服务 **不解析、不校验、不签发** JWT。签名密钥只存在于 Java 侧。
* 只做「原样提取 + 原样透传」：从 FastAPI 请求头取出
  ``Authorization: Bearer <token>``，交给工具调用 Java 后端时放回同一个头。
* 鉴权由 Spring Cloud Gateway 与下游 tenhub-member 的 JwtInterceptor 完成。
* 日志中只打印脱敏指纹，绝不打印完整 token。
"""

from __future__ import annotations

import hashlib
import re

from fastapi import Header, HTTPException, status

# 只允许 RFC 6750 的 Bearer 形式；token 字符集按 JWT/JWS 紧凑序列约束
_AUTH_HEADER_PATTERN = re.compile(r"^Bearer\s+(?P<token>[A-Za-z0-9_\-.]{8,8192})$", re.IGNORECASE)


class MissingTokenError(HTTPException):
    """请求未携带可用 JWT。"""

    def __init__(self, detail: str = "缺少或格式错误的 Authorization 请求头，请先登录") -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def extract_bearer_token(authorization: str | None) -> str:
    """从 ``Authorization`` 头中提取裸 token。

    仅做格式校验，不做任何密码学验证。

    Args:
        authorization: 原始请求头值，形如 ``Bearer eyJhbGciOi...``。

    Returns:
        去掉 ``Bearer `` 前缀后的 token 字符串。

    Raises:
        MissingTokenError: 缺失、前缀错误或 token 明显非法。
    """
    if not authorization or not authorization.strip():
        raise MissingTokenError()

    match = _AUTH_HEADER_PATTERN.match(authorization.strip())
    if not match:
        raise MissingTokenError("Authorization 请求头格式错误，应为 'Bearer <JWT>'")

    return match.group("token")


async def bearer_token_dependency(
    authorization: str | None = Header(
        default=None,
        alias="Authorization",
        description="用户 JWT，形如 'Bearer <token>'；由前端登录后透传，本服务仅原样转发",
    ),
) -> str:
    """FastAPI 依赖：提取并返回原始 JWT。

    作为路由依赖使用，保证未登录请求在进入业务逻辑前即被拒绝。
    """
    return extract_bearer_token(authorization)


def token_fingerprint(token: str) -> str:
    """生成 token 的短指纹，用于日志关联而不泄露凭据。"""
    if not token:
        return "<empty>"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"
