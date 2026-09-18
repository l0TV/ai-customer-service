"""验证 lifespan 预热期间事件循环是否被阻塞。

思路：在 lifespan 里并发发起一个极轻量的 HTTP 请求（GET / ），
若事件循环被阻塞，该请求会被推迟到模型加载结束才能返回。

对比：
* 修复前：懒加载首帧会同步走 ``reload()``（含向量模型加载），请求被卡住约 14s；
* 修复后：``reload()`` 在 ``asyncio.to_thread`` 中执行，请求应立刻返回。

运行::

    python tests/test_startup_nonblocking.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def main() -> int:
    import httpx

    from app.main import app
    from app.rag import hybrid_retriever

    samples: list[float] = []
    startup_done = asyncio.Event()

    async def probe_loop() -> None:
        """在启动过程中反复探测根路径，记录每次耗时。"""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for _ in range(400):
                if startup_done.is_set():
                    return
                start = time.perf_counter()
                try:
                    await client.get("/", timeout=60)
                except Exception:  # noqa: BLE001 - 启动未完成时的异常不影响判定
                    pass
                samples.append(time.perf_counter() - start)
                await asyncio.sleep(0.02)

    mode = "eager" if os.environ.get("AI_CS_EAGER_RELOAD") == "1" else "lazy"

    if mode == "eager":
        # 模拟修复前的写法：工厂在构造时同步 reload（会阻塞事件循环）
        from app.rag.hybrid_retriever import HybridPolicyRetriever

        def eager_factory() -> HybridPolicyRetriever:
            retriever = HybridPolicyRetriever()
            retriever.reload()
            return retriever

        eager_factory = lru_cache(maxsize=1)(eager_factory)
        hybrid_retriever.get_hybrid_retriever_cached = eager_factory  # type: ignore[assignment]
        import app.main as main_module

        main_module.get_hybrid_retriever_cached = eager_factory  # type: ignore[attr-defined]

    probe_task = asyncio.create_task(probe_loop())
    await asyncio.sleep(0.1)

    hybrid_retriever.get_hybrid_retriever_cached.cache_clear()
    start = time.perf_counter()
    async with app.router.lifespan_context(app):
        pass
    lifespan_seconds = time.perf_counter() - start

    startup_done.set()
    await probe_task
    worst = max(samples) if samples else 0.0

    print()
    print(f"MODE={mode}")
    print(f"lifespan 总耗时     : {lifespan_seconds:.2f}s")
    print(f"启动期间探测请求次数 : {len(samples)}")
    print(f"探测请求最大耗时     : {worst:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
