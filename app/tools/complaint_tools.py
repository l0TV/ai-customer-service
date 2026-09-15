"""投诉工单工具：经 Spring Cloud Gateway 调用 Java 后端的投诉接口。

调用链
------
::

    Agent -> create_complaint_ticket(..., runtime)
                    │  从 runtime.config 取 JWT
                    ▼
        httpx  ->  Spring Cloud Gateway (http://127.0.0.1:8080 之外的 :88)
                    │  路由 /api/member/** -> tenhub-member
                    │  Authorization: Bearer <JWT> 原样透传
                    ▼
        ComplaintController  ->  JwtInterceptor 校验 JWT 并解析 userId
                    │
                    ▼
        ums_complaint 落库（userId 一律取自 JWT）

安全边界
--------
* 本服务 **不解析、不校验** JWT，只做原样转发；签名密钥只在 Java 侧。
* 工具的 ``runtime`` 形参由 LangGraph 注入，**不会**出现在模型可见的 tool schema
  中（已由 ``tests/test_agent_tool_passthrough.py`` 实测验证），因此模型既看不到
  JWT，也无法伪造或篡改它。
* **用户身份不从工具入参取**。工具签名里没有 ``user_id``，后端一律以 JWT 中的
  subject 作为工单归属，从根上避免「让 AI 替别人建单」的越权风险。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field

from app.core.config import settings as default_settings
from app.core.logging_config import get_logger
from app.core.request_context import MissingJwtContextError, token_from_runtime_or_args
from app.utils.jwt import token_fingerprint

logger = get_logger(__name__)

# 与 Java 侧 ComplaintController / ComplaintStatusEnum 保持一致
COMPLAINT_TYPES: tuple[str, ...] = (
    "商品质量",
    "物流配送",
    "售后服务",
    "退款问题",
    "发票问题",
    "商家服务态度",
    "其他",
)

# 后端状态码 -> 中文，与 Java 侧 ComplaintStatusEnum 严格对齐
# （取值来自 ums_complaint.status 注释：0-正在受理 1-已返回结果待用户确认 2-已办结）
_STATUS_TEXT = {0: "正在受理", 1: "已返回结果待用户确认", 2: "已办结"}


def _status_text(status: Any) -> str:
    try:
        return _STATUS_TEXT.get(int(status), "正在受理")
    except (TypeError, ValueError):
        return "正在受理"


class ComplaintTicketInput(BaseModel):
    """投诉工单创建入参（模型可见的部分）。"""

    detail: str = Field(
        description=(
            "投诉的具体内容，需包含用户描述的事实：发生了什么、涉及哪个订单或商品、"
            "用户诉求是什么。用简体中文，尽量完整但不要编造用户没说过的信息。"
        ),
        min_length=5,
        max_length=2000,
    )
    order_sn: str = Field(
        default="",
        description=(
            "关联的订单编号。仅在用户明确提供了订单号时填写；"
            "用户未提供时留空字符串，不要猜测或编造订单号。"
        ),
        max_length=64,
    )
    complaint_type: str = Field(
        default="其他",
        description=(
            f"投诉分类，必须从以下选项中选一个：{'、'.join(COMPLAINT_TYPES)}。"
            "无法判断时选「其他」。"
        ),
        max_length=32,
    )
    contact_phone: str = Field(
        default="",
        description="用户留下的联系电话，仅在用户主动提供时填写，否则留空。",
        max_length=32,
    )


def _build_payload(data: ComplaintTicketInput) -> dict[str, Any]:
    """构造发送给 Java 后端的请求体。

    注意：**不包含 userId**。工单归属由后端从 JWT 解析，
    避免客户端伪造身份越权建单。字段名与 Java 侧 ComplaintCreateDto 严格对应。
    """
    return {
        "orderSn": data.order_sn.strip() or None,
        "detail": data.detail.strip(),
        "complaintType": data.complaint_type.strip() or "其他",
        "contactPhone": data.contact_phone.strip() or None,
    }


def _build_headers(jwt_token: str) -> dict[str, str]:
    """构造转发请求头，JWT 原样放入 Authorization。"""
    return {
        # 原样透传：不做任何解码或前缀加工，Gateway/后端自行完成 Bearer 解析与验签
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json",
        # 便于网关侧识别调用来源（日志/限流用，不参与鉴权）
        "X-Caller-Service": default_settings.internal_caller,
    }


def _format_backend_error(status_code: int, body: str) -> str:
    """把后端错误翻译成客服口径的可读提示，不把原始堆栈暴露给用户。"""
    if status_code == 401:
        return (
            "【工单创建失败】用户登录状态无效或已过期（后端返回 401）。"
            "请提示用户重新登录后再试，本次不要谎称工单已创建。"
        )
    if status_code == 403:
        return (
            "【工单创建失败】当前账号无权创建工单（后端返回 403）。"
            "请提示用户联系人工客服处理。"
        )
    if status_code == 400:
        return f"【工单创建失败】提交内容不符合要求（后端返回 400）：{body[:300]}"
    if status_code >= 500:
        return (
            f"【工单创建失败】后端服务异常（HTTP {status_code}），"
            "请告知用户稍后重试或联系人工客服，不要谎称已创建成功。"
        )
    return f"【工单创建失败】HTTP {status_code}：{body[:300]}"


async def _post_to_gateway(payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
    """向网关发起创建请求。"""
    base = default_settings.gateway_base_url.rstrip("/")
    url = f"{base}{default_settings.complaint_create_path}"
    async with httpx.AsyncClient(timeout=default_settings.gateway_timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        return response.status_code, response.text


@tool("create_complaint_ticket", args_schema=ComplaintTicketInput)
async def create_complaint_ticket(
    detail: str,
    runtime: ToolRuntime,
    order_sn: str = "",
    complaint_type: str = "其他",
    contact_phone: str = "",
) -> str:
    """为用户创建投诉工单，工单会提交到平台客服系统由人工跟进。

    当用户表达「投诉 / 举报 / 反馈问题 / 要说法 / 商家态度差 / 商品有问题要追责」
    等诉求，且已说明具体情况时调用本工具。

    调用前请先确认：用户描述的具体问题是什么、是否提供了订单号。
    如果用户只说「我要投诉」而没有说明原因，请先追问具体情况，不要直接调用。

    工单会自动归属到当前登录用户（身份来自其登录凭证，无需也无法指定）。

    Args:
        detail: 投诉的具体内容与诉求。
        runtime: 由 LangGraph 注入的运行时对象，携带用户 JWT；模型不可见。
        order_sn: 关联订单号，用户未提供则留空。
        complaint_type: 投诉分类。
        contact_phone: 联系电话，用户未提供则留空。

    Returns:
        面向模型的执行结果说明，包含工单号或失败原因。
    """
    # --- 1. 取 JWT（本服务唯一的凭据来源，只透传不解析） ---
    try:
        jwt_token = token_from_runtime_or_args(runtime)
    except MissingJwtContextError as exc:
        logger.error("工单工具缺少 JWT 上下文: %s", exc)
        return (
            "【工单创建失败】当前会话缺少用户登录凭证，无法创建工单。"
            "请提示用户先登录后重试。不要谎称工单已创建。"
        )

    data = ComplaintTicketInput(
        detail=detail,
        order_sn=order_sn or "",
        complaint_type=complaint_type or "其他",
        contact_phone=contact_phone or "",
    )
    payload = _build_payload(data)
    headers = _build_headers(jwt_token)

    logger.info(
        "创建投诉工单 -> %s%s (token=%s, type=%s, orderSn=%s)",
        default_settings.gateway_base_url,
        default_settings.complaint_create_path,
        token_fingerprint(jwt_token),
        data.complaint_type,
        data.order_sn or "-",
    )

    # --- 2. 经网关调用 Java 后端 ---
    try:
        status_code, body = await _post_to_gateway(payload, headers)
    except httpx.TimeoutException:
        logger.error("调用网关创建工单超时")
        return (
            "【工单创建失败】调用后端接口超时，工单状态未知。"
            "请告知用户系统繁忙，建议稍后重试或联系人工客服，不要谎称已创建成功。"
        )
    except httpx.HTTPError as exc:
        logger.error("调用网关失败: %s", exc)
        return (
            f"【工单创建失败】无法连接后端服务（{type(exc).__name__}）。"
            "请告知用户稍后重试或联系人工客服。"
        )

    if status_code >= 400:
        logger.warning("创建工单失败 HTTP %s: %s", status_code, body[:300])
        return _format_backend_error(status_code, body)

    # --- 3. 解析后端返回，提取工单号与状态 ---
    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("后端返回非 JSON: %s", body[:200])
        return (
            "【工单创建成功】后端已受理，但返回内容无法解析以获取工单号。"
            "请告知用户投诉已提交，客服会尽快跟进。"
        )

    code = result.get("code")
    if code not in (0, None):
        message = result.get("msg", "未知错误")
        logger.warning("后端业务失败 code=%s msg=%s", code, message)
        return f"【工单创建失败】后端返回业务错误：{message}。请如实告知用户，不要谎称已创建。"

    complaint = result.get("complaint") or {}
    ticket_id = complaint.get("id")
    status_desc = complaint.get("statusDesc") or _status_text(complaint.get("status"))

    logger.info("工单创建成功 id=%s", ticket_id)
    if ticket_id:
        return (
            f"【工单创建成功】工单号：{ticket_id}，当前状态：{status_desc}，"
            f"分类：{data.complaint_type}。\n"
            f"请用客服口吻告知用户：投诉已受理，工单号是 {ticket_id}，"
            "客服会在 1-3 个工作日内跟进处理，并请其保持电话畅通。"
            "不要额外承诺赔偿方案或处理结果。"
        )
    return (
        "【工单创建成功】投诉已提交，客服会尽快跟进。"
        "请告知用户已受理，不要额外承诺赔偿方案或处理结果。"
    )


class ComplaintQueryInput(BaseModel):
    """投诉工单查询入参。"""

    page: int = Field(default=1, description="页码，从 1 开始。", ge=1, le=100)
    limit: int = Field(default=10, description="每页条数。", ge=1, le=50)


@tool("list_my_complaints", args_schema=ComplaintQueryInput)
async def list_my_complaints(
    runtime: ToolRuntime,
    page: int = 1,
    limit: int = 10,
) -> str:
    """查询当前登录用户自己提交过的投诉工单及处理进度。

    当用户询问「我的投诉处理得怎么样了」「我提交的工单在哪看」时调用。
    只能查询当前登录用户本人的工单（后端按 JWT 中的身份过滤，无法越权查询他人）。

    Args:
        runtime: 由 LangGraph 注入的运行时对象，携带用户 JWT；模型不可见。
        page: 页码。
        limit: 每页条数。

    Returns:
        工单列表文本，或失败原因。
    """
    try:
        jwt_token = token_from_runtime_or_args(runtime)
    except MissingJwtContextError:
        return "【查询失败】当前会话缺少用户登录凭证，请提示用户先登录。"

    base = default_settings.gateway_base_url.rstrip("/")
    url = f"{base}{default_settings.complaint_list_path}"
    headers = _build_headers(jwt_token)

    try:
        async with httpx.AsyncClient(timeout=default_settings.gateway_timeout) as client:
            response = await client.get(
                url, params={"page": page, "limit": limit}, headers=headers
            )
    except httpx.HTTPError as exc:
        logger.error("查询工单失败: %s", exc)
        return f"【查询失败】无法连接后端服务（{type(exc).__name__}），请稍后重试。"

    if response.status_code >= 400:
        return _format_backend_error(response.status_code, response.text).replace(
            "工单创建失败", "工单查询失败"
        )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        return "【查询失败】后端返回内容无法解析。"

    if result.get("code") not in (0, None):
        return f"【查询失败】后端返回业务错误：{result.get('msg', '未知错误')}"

    page_data = result.get("page") or {}
    records = page_data.get("list") or []
    if not records:
        return "【查询结果】该用户目前没有任何投诉工单记录。请如实告知用户。"

    lines: list[str] = [f"【查询结果】共 {page_data.get('totalCount', len(records))} 条工单："]
    for item in records:
        status_desc = item.get("statusDesc") or _status_text(item.get("status"))
        parts = [
            f"- 工单号 {item.get('id')}",
            f"分类 {item.get('complaintType') or '其他'}",
            f"状态 {status_desc}",
            f"提交时间 {item.get('createTime') or '未知'}",
            f"内容：{str(item.get('detail') or '')[:80]}",
        ]
        if item.get("response"):
            parts.append(f"客服回复：{str(item.get('response'))[:80]}")
        lines.append("｜".join(parts))
    lines.append("请用客服口吻汇总告知用户，不要编造列表中没有的工单或处理结果。")
    return "\n".join(lines)


def get_complaint_tools() -> list[Any]:
    """返回投诉类工具列表。"""
    return [create_complaint_ticket, list_my_complaints]
