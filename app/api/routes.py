"""FastAPI 路由。

JWT 透传的**唯一入口**就在这里：
1. 依赖 ``bearer_token_dependency`` 从请求头提取原始 JWT（不解析、不校验）；
2. 用 ``build_run_config`` 把 JWT 放进 ``RunnableConfig``；
3. 后续 Agent 与工具全链路由 LangChain 传递该 config，工具调用 Java 后端时
   原样放回 ``Authorization: Bearer <JWT>``。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.agent.customer_service_agent import get_agent, normalize_session_id
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ComponentStatus,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceRef,
    ToolStep,
)
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.request_context import build_run_config
from app.rag.rag_chain import get_rag_chain
from app.utils.jwt import bearer_token_dependency, token_fingerprint

logger = get_logger(__name__)

router = APIRouter()


# ----------------------------------------------------------------------
# 对话
# ----------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="智能客服对话",
    description=(
        "统一入口。mode=auto 时由 Agent 自主决定：政策类问题走 RAG 混合检索问答，"
        "投诉类诉求走工单创建工具。需要携带用户 JWT。"
    ),
)
async def chat(
    payload: ChatRequest,
    jwt_token: str = Depends(bearer_token_dependency),
) -> ChatResponse:
    """处理一轮用户对话。"""
    session_id = normalize_session_id(
        payload.session_id, fallback=f"anon-{uuid.uuid4().hex[:16]}"
    )
    message = payload.message.strip()

    logger.info(
        "收到对话请求 session=%s mode=%s token=%s len=%d",
        session_id,
        payload.mode,
        token_fingerprint(jwt_token),
        len(message),
    )

    config = build_run_config(
        jwt_token,
        session_id=session_id,
        user_tag=token_fingerprint(jwt_token),
    )

    # --- 强制 RAG 模式：只做政策问答，不触碰工单接口 ---
    if payload.mode == "rag":
        result = await get_rag_chain().aanswer(
            message, top_k=payload.top_k or None
        )
        return ChatResponse(
            answer=result.answer,
            mode="rag",
            session_id=session_id,
            answerable=result.answerable,
            sources=[SourceRef(**source) for source in result.sources],
            reason=result.reason,
        )

    # --- auto / agent：交给 Agent 自主决策 ---
    try:
        agent_result = await get_agent().ainvoke(
            message, config=config, session_id=session_id
        )
    except ValueError as exc:
        # 最常见的 ValueError 是未配置大模型密钥；把它翻译成运维能直接照做的提示
        if not settings.llm_configured:
            logger.error("对话失败：未配置大模型密钥")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "智能客服未配置大模型密钥，对话功能不可用。"
                    "请在 ai-customer-service/.env 中设置 AI_CS_DEEPSEEK_API_KEY 后重启服务。"
                ),
            ) from exc
        logger.exception("Agent 参数错误")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"智能客服处理失败：{exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 对外只暴露可读错误
        logger.exception("Agent 执行失败")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"智能客服处理失败：{type(exc).__name__}",
        ) from exc

    return ChatResponse(
        answer=agent_result.answer,
        mode=payload.mode,
        session_id=session_id,
        steps=[ToolStep(**step.to_dict()) for step in agent_result.steps],
        used_tools=agent_result.used_tools,
        reason="agent",
    )


@router.post(
    "/chat/stream",
    summary="智能客服对话（SSE 流式）",
    description="以 Server-Sent Events 推送 tool_start / token / done 事件。",
)
async def chat_stream(
    payload: ChatRequest,
    jwt_token: str = Depends(bearer_token_dependency),
) -> StreamingResponse:
    """SSE 流式对话。"""
    session_id = normalize_session_id(
        payload.session_id, fallback=f"anon-{uuid.uuid4().hex[:16]}"
    )
    config = build_run_config(
        jwt_token,
        session_id=session_id,
        user_tag=token_fingerprint(jwt_token),
    )
    message = payload.message.strip()

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in get_agent().astream_events(
                message, config=config, session_id=session_id
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 - 流中异常需以事件形式告知前端
            logger.exception("流式对话失败")
            error_event = {
                "type": "error",
                "message": f"智能客服处理失败：{type(exc).__name__}",
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关闭 Nginx 缓冲，保证 token 实时下发
            "X-Accel-Buffering": "no",
        },
    )


# ----------------------------------------------------------------------
# 知识库
# ----------------------------------------------------------------------
@router.post(
    "/rag/query",
    response_model=ChatResponse,
    summary="纯政策问答（不经 Agent）",
    description="调试与内部调用用：直接执行混合检索 + 受约束生成。",
)
async def rag_query(
    payload: ChatRequest,
    jwt_token: str = Depends(bearer_token_dependency),
) -> ChatResponse:
    """只走 RAG，不调用任何写接口。"""
    session_id = normalize_session_id(
        payload.session_id, fallback=f"rag-{uuid.uuid4().hex[:16]}"
    )
    result = await get_rag_chain().aanswer(
        payload.message.strip(), top_k=payload.top_k or None
    )
    return ChatResponse(
        answer=result.answer,
        mode="rag",
        session_id=session_id,
        answerable=result.answerable,
        sources=[SourceRef(**source) for source in result.sources],
        reason=result.reason,
    )


@router.get(
    "/rag/search",
    summary="混合检索调试接口",
    description="只返回检索结果（BM25 + 向量 + RRF 融合），不调用大模型。",
)
async def rag_search(
    q: str = Query(..., min_length=1, max_length=500, description="查询语句"),
    top_k: int = Query(default=0, ge=0, le=50),
    _: str = Depends(bearer_token_dependency),
) -> dict[str, Any]:
    """查看混合检索命中了哪些片段及其相似度，用于排查召回质量。"""
    from app.rag.hybrid_retriever import get_hybrid_retriever

    retriever = get_hybrid_retriever()
    scored = retriever.retrieve_with_scores(q, top_k=top_k or None)
    return {
        "code": 0,
        "msg": "success",
        "query": q,
        "corpus_size": retriever.corpus_size,
        "threshold": settings.relevance_threshold,
        "results": [
            {
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "source": doc.metadata.get("source", ""),
                "section": doc.metadata.get("section", ""),
                "score": round(score, 4),
                "passed_threshold": score >= settings.relevance_threshold,
                "content": doc.page_content[:400],
            }
            for doc, score in scored
        ],
    }


@router.post(
    "/rag/ingest",
    response_model=IngestResponse,
    summary="政策文档入库",
    description=(
        "解析 data/policies 下的 PDF/Word/Markdown/TXT，分块后写入 Milvus，"
        "并同步刷新 BM25 语料快照。首次入库或更换向量模型时请置 rebuild=true。"
    ),
)
async def rag_ingest(
    payload: IngestRequest,
    _: str = Depends(bearer_token_dependency),
) -> IngestResponse:
    """执行知识库入库。"""
    from app.rag.ingest import ingest_policies

    try:
        stats = await ingest_policies(rebuild=payload.rebuild, directory=payload.directory)
    except Exception as exc:  # noqa: BLE001 - 入库失败需返回可读原因
        logger.exception("知识库入库失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"入库失败：{type(exc).__name__}: {exc}",
        ) from exc

    return IngestResponse(**stats)


@router.post(
    "/rag/upload",
    response_model=IngestResponse,
    summary="上传单个政策文档并入库",
    description="支持 .md/.txt/.pdf/.docx，文件保存到 data/policies 后立即入库。",
)
async def rag_upload(
    file: UploadFile = File(..., description="政策文档"),
    rebuild: bool = Query(default=False, description="是否全量重建集合"),
    _: str = Depends(bearer_token_dependency),
) -> IngestResponse:
    """上传并入库单个文档。"""
    from app.rag.ingest import ingest_uploaded_file

    filename = file.filename or "unnamed"
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in settings.supported_suffixes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型 {suffix}，仅支持 {list(settings.supported_suffixes)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="上传文件为空")

    try:
        stats = await ingest_uploaded_file(filename, content, rebuild=rebuild)
    except Exception as exc:  # noqa: BLE001
        logger.exception("上传入库失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传入库失败：{type(exc).__name__}: {exc}",
        ) from exc

    return IngestResponse(**stats)


# ----------------------------------------------------------------------
# 运维
# ----------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    """探测 Milvus、向量模型、BM25 语料、大模型配置与网关可达性。"""
    components: list[ComponentStatus] = []

    # Milvus
    try:
        from app.rag.vectorstore import get_vector_store

        count = get_vector_store().count()
        components.append(
            ComponentStatus(
                name="milvus",
                healthy=True,
                detail=f"{settings.milvus_uri} / 集合 {settings.milvus_collection} / {count} 条",
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(
            ComponentStatus(name="milvus", healthy=False, detail=f"{type(exc).__name__}: {exc}")
        )

    # 向量模型（只检查是否已加载，避免健康检查触发长时间加载）
    try:
        from app.rag.embeddings import get_embeddings

        embeddings = get_embeddings()
        loaded = embeddings._model is not None  # noqa: SLF001 - 仅探测状态
        components.append(
            ComponentStatus(
                name="embedding",
                healthy=True,
                detail=(
                    f"{settings.embedding_model}（已加载）"
                    if loaded
                    else f"{settings.embedding_model}（首次调用时加载）"
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(
            ComponentStatus(name="embedding", healthy=False, detail=f"{type(exc).__name__}: {exc}")
        )

    # BM25 语料
    corpus_path = settings.bm25_corpus_path
    components.append(
        ComponentStatus(
            name="bm25_corpus",
            healthy=corpus_path.exists(),
            detail=f"{corpus_path} ({'存在' if corpus_path.exists() else '缺失，请先入库'})",
        )
    )

    # 大模型配置
    components.append(
        ComponentStatus(
            name="llm",
            healthy=settings.llm_configured,
            detail=(
                f"{settings.chat_model} @ {settings.deepseek_base_url}"
                if settings.llm_configured
                else "未配置 AI_CS_DEEPSEEK_API_KEY"
            ),
        )
    )

    # 网关
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.get(settings.gateway_base_url)
        components.append(
            ComponentStatus(name="gateway", healthy=True, detail=settings.gateway_base_url)
        )
    except Exception as exc:  # noqa: BLE001 - 网关未启动不算服务不可用
        components.append(
            ComponentStatus(
                name="gateway",
                healthy=False,
                detail=f"{settings.gateway_base_url} 不可达（{type(exc).__name__}）；工单功能将不可用",
            )
        )

    # Nacos 服务注册
    from app.core.nacos_registry import get_registrar

    registration = get_registrar().registration
    if not settings.nacos_enabled:
        components.append(
            ComponentStatus(name="nacos", healthy=True, detail="已通过配置关闭注册")
        )
    elif registration is None:
        components.append(
            ComponentStatus(name="nacos", healthy=False, detail="尚未执行注册（服务可能未完成启动）")
        )
    else:
        components.append(
            ComponentStatus(
                name="nacos",
                healthy=registration.registered,
                detail=registration.describe(),
            )
        )

    critical = {"milvus", "embedding", "llm"}
    overall = "ok" if all(c.healthy for c in components if c.name in critical) else "degraded"
    return HealthResponse(status=overall, app=settings.app_name, components=components)


@router.get("/config", summary="查看当前生效配置（脱敏）")
async def show_config() -> dict[str, Any]:
    """输出生效配置，便于联调排查。API Key 仅显示是否已配置。"""
    return {
        "code": 0,
        "msg": "success",
        "config": {
            "chat_model": settings.chat_model,
            "llm_base_url": settings.deepseek_base_url,
            "llm_key_configured": settings.llm_configured,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "milvus_uri": settings.milvus_uri,
            "milvus_collection": settings.milvus_collection,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "dense_top_k": settings.dense_top_k,
            "sparse_top_k": settings.sparse_top_k,
            "fusion_top_k": settings.fusion_top_k,
            "rrf_c": settings.rrf_c,
            "rrf_weights": [settings.rrf_weight_dense, settings.rrf_weight_sparse],
            "relevance_threshold": settings.relevance_threshold,
            "gateway_base_url": settings.gateway_base_url,
            "complaint_create_path": settings.complaint_create_path,
            "policy_dir": str(settings.policy_dir),
        },
    }
