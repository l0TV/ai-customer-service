"""生成与 Java 侧 JwtUtil 兼容的测试 JWT。

**仅用于本地联调测试**。真正的签名密钥只存在于 Nacos 配置中，
本脚本从环境变量读取，不硬编码任何密钥。

生成算法与 `com.tv10.tenhub.member.util.JwtUtil#generateTokenWithUserType` 对齐：
  * 签名算法 HS256，密钥为 tenhub.jwt.secret 的 UTF-8 字节
  * `sub`     = 用户 id
  * `userType`= user / admin（AuthConstant.USER_TYPE）
  * `iat` / `exp`

用法::

    $env:TENHUB_JWT_SECRET = "<从 Nacos 取的 tenhub.jwt.secret>"
    python scripts/make_test_jwt.py --user-id 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="生成联调用 JWT")
    parser.add_argument("--user-id", default="1", help="用户 id（写入 sub）")
    parser.add_argument("--user-type", default="user", choices=["user", "admin"])
    parser.add_argument("--minutes", type=int, default=120, help="有效期（分钟）")
    parser.add_argument(
        "--secret",
        default=os.environ.get("TENHUB_JWT_SECRET", ""),
        help="签名密钥；默认读环境变量 TENHUB_JWT_SECRET",
    )
    parser.add_argument(
        "--from-nacos",
        action="store_true",
        help="从本地 Nacos 拉取 tenhub-member.yml 中的密钥（仅联调用）",
    )
    args = parser.parse_args()

    secret = args.secret.strip()
    if not secret and args.from_nacos:
        import httpx
        import yaml

        url = (
            "http://127.0.0.1:8848/nacos/v1/cs/configs"
            "?dataId=tenhub-member.yml&group=dev"
            "&tenant=2c349bea-d121-4bf6-ad7d-b9dab34c80fa"
        )
        config = yaml.safe_load(httpx.get(url, timeout=10).text)
        secret = (config.get("tenhub", {}).get("jwt", {}) or {}).get("secret", "")

    if not secret:
        print(
            "缺少签名密钥。请设置 TENHUB_JWT_SECRET 环境变量，或加 --from-nacos。",
            file=sys.stderr,
        )
        return 1

    try:
        import jwt
    except ImportError:
        print("需要 PyJWT：pip install pyjwt", file=sys.stderr)
        return 1

    now = int(time.time())
    payload = {
        "sub": str(args.user_id),
        "userType": args.user_type,
        "iat": now,
        "exp": now + args.minutes * 60,
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    # 只输出 token 本身，便于脚本直接取值
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
