"""经网关验证 AI 服务是否可被 Spring Cloud Gateway 发现并路由。

验证点：
1. 网关能否通过 Nacos 发现 tenhub-ai-service 并把 /api/ai/** 路由过去；
2. 网关路由未与 renren-fast 的 /api/** 兜底路由冲突；
3. 经网关调用工单接口时，JWT 仍能原样透传到 Java 后端。

用法::

    python scripts/verify_gateway_ai.py --jwt <token>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

GATEWAY = "http://127.0.0.1:88"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jwt", required=True, help="用户 JWT")
    parser.add_argument("--gateway", default=GATEWAY)
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.jwt}"}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # ---------- 1) 网关 -> AI 服务健康检查 ----------
    print("--- 1) 经网关访问 AI 服务 /api/ai/health ---")
    try:
        resp = httpx.get(f"{args.gateway}/api/ai/health", headers=headers, timeout=60)
        check("网关成功路由到 AI 服务", resp.status_code == 200, f"HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"     overall = {data.get('status')}")
            for component in data.get("components", []):
                print(
                    f"     {component['name']:<12} {str(component['healthy']):<6} "
                    f"{component['detail']}"
                )
            names = {c["name"]: c["healthy"] for c in data.get("components", [])}
            check("组件中包含 nacos 项", "nacos" in names)
            check("Nacos 注册状态为健康", names.get("nacos", False))
        else:
            print(f"     body = {resp.text[:400]}")
    except Exception as exc:  # noqa: BLE001
        check("网关成功路由到 AI 服务", False, f"{type(exc).__name__}: {exc}")

    # ---------- 2) 确认未被 renren-fast 兜底路由抢走 ----------
    print("\n--- 2) 确认未与 renren-fast 的 /api/** 兜底路由冲突 ---")
    try:
        # renren-fast 对未知路径会返回它自己的 404/401，而 AI 服务返回的是 AI 的 JSON。
        # 这里用 /api/ai/config 判断响应体是否来自 AI 服务。
        resp = httpx.get(f"{args.gateway}/api/ai/config", headers=headers, timeout=40)
        is_ai = False
        if resp.status_code == 200:
            body = resp.json()
            is_ai = "config" in body and "embedding_model" in body.get("config", {})
        check(
            "路由命中的是 AI 服务而非 renren-fast",
            is_ai,
            f"HTTP {resp.status_code} body={resp.text[:120]}",
        )
    except Exception as exc:  # noqa: BLE001
        check("路由命中的是 AI 服务而非 renren-fast", False, f"{type(exc).__name__}: {exc}")

    # ---------- 3) 经网关调用 AI 的政策检索 ----------
    print("\n--- 3) 经网关调用 AI 混合检索 ---")
    try:
        resp = httpx.get(
            f"{args.gateway}/api/ai/rag/search",
            params={"q": "退货运费谁承担", "top_k": 3},
            headers=headers,
            timeout=90,
        )
        ok = resp.status_code == 200 and resp.json().get("results")
        detail = ""
        if resp.status_code == 200:
            results = resp.json()["results"]
            detail = f"命中 {len(results)} 条，首条={results[0]['source']} score={results[0]['score']}"
        check("经网关检索成功", bool(ok), detail or f"HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        check("经网关检索成功", False, f"{type(exc).__name__}: {exc}")

    # ---------- 4) 经网关创建工单（验证 JWT 二次透传）----------
    print("\n--- 4) 经网关创建投诉工单（JWT 透传验证）---")
    try:
        resp = httpx.post(
            f"{args.gateway}/api/member/complaint/create",
            json={
                "orderSn": "GW-AI-VERIFY-001",
                "detail": "网关+AI联调验证：经网关访问 Java 后端创建工单，验证 JWT 透传链路。",
                "complaintType": "其他",
            },
            headers=headers,
            timeout=40,
        )
        data = resp.json() if resp.status_code == 200 else {}
        ticket = (data.get("complaint") or {}).get("id")
        check(
            "经网关创建工单成功（JWT 透传正常）",
            resp.status_code == 200 and data.get("code") == 0,
            f"HTTP {resp.status_code} ticket={ticket}",
        )
        check("工单状态为新的三态模型", (data.get("complaint") or {}).get("statusDesc") == "正在受理",
              str((data.get("complaint") or {}).get("statusDesc")))
    except Exception as exc:  # noqa: BLE001
        check("经网关创建工单成功（JWT 透传正常）", False, f"{type(exc).__name__}: {exc}")

    # ---------- 5) 无 JWT 应被拒 ----------
    print("\n--- 5) 经网关无 JWT 访问 AI 服务 ---")
    try:
        resp = httpx.post(
            f"{args.gateway}/api/ai/chat", json={"message": "退货政策"}, timeout=30
        )
        check("无 JWT 时 AI 服务返回 401", resp.status_code == 401, f"HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        check("无 JWT 时 AI 服务返回 401", False, f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 66)
    print(f"共 6 项检查，失败 {len(failures)} 项")
    for name in failures:
        print(f"  - {name}")
    print("=" * 66)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
