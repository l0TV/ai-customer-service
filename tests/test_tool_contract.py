"""工单工具 HTTP 契约测试。

目的：在不启动完整 Java 服务的前提下，用一个本地假网关（http.server）接收
工单工具发出的真实 HTTP 请求，逐项校验**契约细节**：

1. 请求方法与路径是否正确（`POST /api/member/complaint/create`）；
2. `Authorization` 是否为 `Bearer <原样JWT>`——即确实只透传、未做任何加工；
3. 请求体字段名是否与 Java 侧 `ComplaintCreateDto` 严格对应；
4. 请求体中**绝不能出现 userId**（否则越权风险）；
5. 后端返回成功时，工具是否正确回传工单号；
6. 后端返回 401/500/业务错误码时，工具是否如实报错而非谎报成功。

运行::

    python tests/test_tool_contract.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAPTURED: dict[str, object] = {}
RESPONSE: dict[str, object] = {}


class FakeGatewayHandler(BaseHTTPRequestHandler):
    """按 RESPONSE 中的脚本返回响应，并记录收到的请求。"""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # noqa: D102 - 静默
        return

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""

        CAPTURED["method"] = method
        CAPTURED["path"] = self.path
        CAPTURED["authorization"] = self.headers.get("Authorization")
        CAPTURED["content_type"] = self.headers.get("Content-Type")
        CAPTURED["caller"] = self.headers.get("X-Caller-Service")
        try:
            CAPTURED["body"] = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            CAPTURED["body"] = raw

        status = int(RESPONSE.get("status", 200))
        payload = RESPONSE.get("payload", {})
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json;charset=UTF-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")


def start_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), FakeGatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def build_runtime(jwt: str | None):
    from langgraph.prebuilt import ToolRuntime

    configurable = {"jwt_token": jwt} if jwt else {}
    return ToolRuntime(
        state={},
        context=None,
        config={"configurable": configurable},
        stream_writer=lambda *_a, **_k: None,
        tool_call_id="call_contract",
        store=None,
        tools=[],
    )


def main() -> int:
    from app.core.config import settings
    from app.tools.complaint_tools import create_complaint_ticket

    server, port = start_server()
    # 把网关地址指向假服务
    object.__setattr__(settings, "gateway_base_url", f"http://127.0.0.1:{port}")

    JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI3NzcifQ.CONTRACTSIG"
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))
        print(f"{'PASS' if condition else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))

    try:
        # ---------- 场景 1：后端成功 ----------
        print("--- 场景 1：后端创建成功 ---")
        CAPTURED.clear()
        RESPONSE.clear()
        RESPONSE.update(
            {
                "status": 200,
                "payload": {
                    "code": 0,
                    "msg": "success",
                    "complaint": {
                        "id": 12345,
                        "status": 0,
                        "statusDesc": "正在受理",
                        "complaintType": "物流配送",
                    },
                },
            }
        )

        text = asyncio.run(
            create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
                detail="收到的商品外包装破损，申请换货。",
                runtime=build_runtime(JWT),
                order_sn="SN202501010001",
                complaint_type="物流配送",
                contact_phone="13800138000",
            )
        )
        print(f"     工具返回：{text[:160]}")

        check("请求方法为 POST", CAPTURED.get("method") == "POST", str(CAPTURED.get("method")))
        check(
            "请求路径与网关路由一致",
            CAPTURED.get("path") == "/api/member/complaint/create",
            str(CAPTURED.get("path")),
        )
        check(
            "Authorization 原样透传（Bearer + 未改动的 JWT）",
            CAPTURED.get("authorization") == f"Bearer {JWT}",
            str(CAPTURED.get("authorization"))[:60],
        )

        body = CAPTURED.get("body")
        check("请求体是 JSON 对象", isinstance(body, dict), type(body).__name__)
        if isinstance(body, dict):
            check(
                "字段名与 ComplaintCreateDto 一致",
                set(body) == {"orderSn", "detail", "complaintType", "contactPhone"},
                str(sorted(body)),
            )
            check(
                "请求体绝不包含 userId（防越权）",
                "userId" not in body and "user_id" not in body,
                str(sorted(body)),
            )
            check("orderSn 正确", body.get("orderSn") == "SN202501010001", str(body.get("orderSn")))
            check(
                "detail 正确",
                body.get("detail") == "收到的商品外包装破损，申请换货。",
                str(body.get("detail")),
            )
            check(
                "complaintType 正确",
                body.get("complaintType") == "物流配送",
                str(body.get("complaintType")),
            )

        check(
            "成功时回传工单号",
            "12345" in str(text) and "【工单创建成功】" in str(text),
            str(text)[:100],
        )

        # ---------- 场景 2：401 ----------
        print("\n--- 场景 2：后端返回 401（JWT 失效）---")
        RESPONSE.update({"status": 401, "payload": {"code": 401, "msg": "Token 已过期"}})
        text = asyncio.run(
            create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
                detail="测试 401 场景的投诉内容。",
                runtime=build_runtime(JWT),
            )
        )
        check(
            "401 时如实报错且不谎报成功",
            "【工单创建失败】" in text and "重新登录" in text and "成功" not in text.replace(
                "【工单创建失败】", ""
            ),
            text[:100],
        )

        # ---------- 场景 3：业务错误码 ----------
        print("\n--- 场景 3：后端返回业务错误码 ---")
        RESPONSE.update(
            {"status": 200, "payload": {"code": 400, "msg": "投诉内容过长，请控制在 2000 字以内"}}
        )
        text = asyncio.run(
            create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
                detail="测试业务错误码场景的投诉内容。",
                runtime=build_runtime(JWT),
            )
        )
        check(
            "业务错误码时如实转达后端消息",
            "【工单创建失败】" in text and "2000" in text,
            text[:120],
        )

        # ---------- 场景 4：500 ----------
        print("\n--- 场景 4：后端 500 ---")
        RESPONSE.update({"status": 500, "payload": {"code": 500, "msg": "内部错误"}})
        text = asyncio.run(
            create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
                detail="测试 500 场景的投诉内容。",
                runtime=build_runtime(JWT),
            )
        )
        check(
            "500 时如实报错",
            "【工单创建失败】" in text and "异常" in text,
            text[:120],
        )

        # ---------- 场景 5：缺少 JWT ----------
        print("\n--- 场景 5：缺少 JWT ---")
        CAPTURED.clear()
        text = asyncio.run(
            create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
                detail="测试缺少凭据时不发请求。",
                runtime=build_runtime(None),
            )
        )
        check(
            "缺少 JWT 时拒绝执行",
            "缺少用户登录凭证" in text,
            text[:100],
        )
        check(
            "缺少 JWT 时未向后端发起请求",
            not CAPTURED,
            f"captured={list(CAPTURED)}",
        )

    finally:
        server.shutdown()

    print("\n" + "=" * 60)
    failed = [name for name, ok, _ in results if not ok]
    print(f"共 {len(results)} 项检查，失败 {len(failed)} 项")
    for name in failed:
        detail = next(d for n, ok, d in results if n == name and not ok)
        print(f"  - {name} | {detail}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
