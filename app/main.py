"""FastAPI 应用入口。

启动（在项目根路径下）：
    uvicorn app.main:app --host 0.0.0.0 --port 8090

或直接运行：
    python -m app.main
"""

from __future__ import annotations

import contextlib
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging

setup_logging(settings.log_level)
logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """启动/关闭钩子。"""
    logger.info("=" * 72)
    logger.info("拾汇商城智能客服服务启动中 v%s", __version__)
    logger.info("对话模型     : %s @ %s", settings.chat_model, settings.deepseek_base_url)
    logger.info(
        "大模型密钥   : %s", "已配置" if settings.llm_configured else "未配置（对话将不可用）"
    )
    logger.info("向量模型     : %s (%d 维)", settings.embedding_model, settings.embedding_dim)
    logger.info("Milvus       : %s / %s", settings.milvus_uri, settings.milvus_collection)
    logger.info("网关地址     : %s", settings.gateway_base_url)
    logger.info("政策文档目录 : %s", settings.policy_dir)
    logger.info("=" * 72)

    # 注册到 Nacos，使网关可通过 lb://tenhub-ai-service 发现本服务
    from app.core.nacos_registry import get_registrar

    registrar = get_registrar()
    registration = await registrar.register()
    if registration.registered:
        logger.info("Nacos        : %s", registration.describe())
    else:
        logger.warning("Nacos        : %s", registration.describe())

    # 预热混合检索器：BM25 语料载入 + Milvus 检索器构建
    try:
        from app.rag.hybrid_retriever import get_hybrid_retriever_cached

        retriever = get_hybrid_retriever_cached()
        built = await _warmup(retriever)
        if not built:
            logger.warning(
                "混合检索未完全就绪或 BM25 语料缺失。"
                "请调用 POST /api/ai/rag/ingest 完成知识库入库。"
            )
    except Exception as exc:  # noqa: BLE001 - 预热失败不应阻止服务启动
        logger.error("预热检索器失败（服务仍会启动）: %s", exc)

    yield

    # 关闭：先从 Nacos 注销，避免网关继续把流量路由到已停止的实例
    await registrar.deregister()
    logger.info("拾汇商城智能客服服务已停止")


async def _warmup(retriever) -> bool:
    """在线程池中预热检索器，避免阻塞事件循环。"""
    import asyncio

    def _do() -> bool:
        return retriever.reload()

    built = await asyncio.to_thread(_do)
    logger.info(
        "检索器预热完成：BM25 语料 %d 条，模式=%s",
        retriever.corpus_size,
        "BM25+向量混合" if built else "纯向量",
    )
    return built


app = FastAPI(
    title="拾汇商城 AI 智能客服服务",
    description=(
        "基于 LangChain + Milvus 的智能客服：\n\n"
        "* **RAG 政策问答**：BM25 + 向量混合检索，RRF 融合，严格基于检索上下文，无法回答时拒答\n"
        "* **Agent 自主决策**：政策咨询走知识库检索，投诉诉求走工单创建\n"
        "* **JWT 透传**：本服务不解析不校验 JWT，仅经 RunnableConfig 透传，"
        "工具调用时原样放入 Authorization 头，经 Spring Cloud Gateway 转发给 Java 后端\n"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 允许前端与网关跨域调用；生产环境应把 allow_origins 收敛到具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/ai", tags=["智能客服"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """统一异常出口，避免把内部堆栈返回给调用方。"""
    logger.exception("未处理异常 %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "msg": f"服务内部错误：{type(exc).__name__}", "answer": ""},
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/api/ai/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
