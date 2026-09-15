"""端到端测试：Python 工单工具 -> 真实 Java 后端。

前置条件（缺一不可）:
  * tenhub-member 已在 127.0.0.1:10000 运行
  * MySQL 中已存在 ums_complaint 表（含 complaint_type / contact_phone 列）
  * 能签名出合法 JWT（见 scripts/make_test_jwt.py）

说明：本机 Gateway(:88) 未启动，因此把工具的目标地址直接指向 member 服务。
JWT 透传链路、请求体契约、后端解析 userId、落库全流程与经网关时**完全一致**，
差别仅在于少了一跳路由转发。

运行::

    $env:TENHUB_JWT_SECRET = "<secret>"   # 或让脚本用 --from-nacos
    python tests/test_e2e_live.py --from-nacos
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


def make_jwt(user_id: str, from_nacos: bool) -> str:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "make_test_jwt.py"), "--user-id", user_id]
    if from_nacos:
        cmd.append("--from-nacos")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"生成 JWT 失败: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-nacos", action="store_true", help="从 Nacos 取签名密钥")
    parser.add_argument("--member-url", default="http://127.0.0.1:10000")
    parser.add_argument("--user-id", default="1")
    args = parser.parse_args()

    from app.core.config import settings
    from app.tools.complaint_tools import create_complaint_ticket, list_my_complaints
    from langgraph.prebuilt import ToolRuntime

    # 指向真实 member 服务（本机网关未启动；生产环境用网关地址）
    object.__setattr__(settings, "gateway_base_url", args.member_url)
    object.__setattr__(settings, "complaint_create_path", "/member/complaint/create")
    object.__setattr__(settings, "complaint_list_path", "/member/complaint/list")

    print("=" * 70)
    print(f"E2E：Python 工具 -> {args.member_url}")
    print("=" * 70)

    jwt = make_jwt(args.user_id, args.from_nacos)
    check("生成合法 JWT", bool(jwt) and jwt.count(".") == 2, f"{jwt[:32]}…")

    def runtime(token: str | None):
        return ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"jwt_token": token} if token else {}},
            stream_writer=lambda *_a, **_k: None,
            tool_call_id="e2e",
            store=None,
            tools=[],
        )

    # ---------- 1. 创建工单（真实落库） ----------
    print("\n--- 1) 创建投诉工单 ---")
    detail = "端到端测试：订单商品屏幕碎裂，商家拒绝退货，申请平台介入并补偿运费。"
    text = asyncio.run(
        create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
            detail=detail,
            runtime=runtime(jwt),
            order_sn="E2E-LIVE-0001",
            complaint_type="商品质量",
            contact_phone="13900139000",
        )
    )
    print(f"     工具返回：{text}")

    check("工单创建成功", "【工单创建成功】" in text, text[:80])

    # 提取工单号
    ticket_id = ""
    for marker in ("工单号：", "工单号是 "):
        if marker in text:
            tail = text.split(marker, 1)[1]
            ticket_id = "".join(ch for ch in tail.split("，")[0].split("。")[0] if ch.isdigit())
            break
    check("返回了工单号", bool(ticket_id), f"ticket_id={ticket_id}")
    check("状态文案为新的三态模型", "正在受理" in text, text[:80])

    # ---------- 2. 查询自己的工单 ----------
    print("\n--- 2) 查询我的投诉工单 ---")
    listed = asyncio.run(
        list_my_complaints.coroutine(runtime=runtime(jwt), page=1, limit=10)  # type: ignore[attr-defined]
    )
    print(f"     工具返回：{str(listed)[:300]}")
    check("查询到工单且包含本次工单号", ticket_id and ticket_id in str(listed), f"ticket={ticket_id}")

    # ---------- 3. 无 JWT 必须被拒 ----------
    print("\n--- 3) 无 JWT 场景 ---")
    denied = asyncio.run(
        create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
            detail="无凭据时不应创建任何工单。",
            runtime=runtime(None),
        )
    )
    check("无 JWT 时工具层拒绝", "缺少用户登录凭证" in str(denied), str(denied)[:80])

    # ---------- 4. 伪造 JWT 必须被后端拒绝 ----------
    print("\n--- 4) 伪造 JWT 场景（后端应鉴权失败）---")
    forged = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5OTk5OTkifQ.FORGED_SIGNATURE_XX"
    forged_result = asyncio.run(
        create_complaint_ticket.coroutine(  # type: ignore[attr-defined]
            detail="伪造签名时后端必须拒绝，不能产生工单。",
            runtime=runtime(forged),
            complaint_type="其他",
        )
    )
    print(f"     工具返回：{forged_result[:200]}")
    check(
        "伪造签名被后端拒绝（未谎报成功）",
        "【工单创建失败】" in str(forged_result),
        str(forged_result)[:80],
    )

    # ---------- 5. 校验落库的 user_id 来自 JWT ----------
    print("\n--- 5) 校验落库数据的 user_id 来自 JWT ---")
    try:
        import httpx

        # 通过 list 接口间接验证：只有 user_id 匹配才查得到
        resp = httpx.get(
            f"{args.member_url}/member/complaint/list",
            params={"page": 1, "limit": 20},
            headers={"Authorization": f"Bearer {jwt}"},
            timeout=15,
        )
        data = resp.json()
        records = (data.get("page") or {}).get("list") or []
        ids = [str(item.get("id")) for item in records]
        check(
            "JWT 身份可查到刚创建的工单（说明 user_id 取自 JWT）",
            ticket_id in ids,
            f"ticket={ticket_id} visible_ids={ids}",
        )
        check("返回体未泄露 user_id 字段", all("userId" not in item for item in records))
        if records:
            phone = records[0].get("contactPhone") or ""
            check("联系电话已脱敏", "****" in phone or phone == "", f"phone={phone}")
    except Exception as exc:  # noqa: BLE001
        check("校验落库数据", False, f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"共 {len(RESULTS)} 项检查，失败 {len(failed)} 项")
    for name in failed:
        detail_text = next(d for n, ok, d in RESULTS if n == name and not ok)
        print(f"  - {name} | {detail_text}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
