"""API 请求/响应契约。

统一以 ``code``/``msg`` 风格对齐 Java 侧的 ``R`` 结构，便于前端复用同一套解包逻辑。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 对话路由模式：auto 交给 Agent 自主决策，rag 强制走政策问答
ChatMode = Literal["auto", "rag", "agent"]


# ----------------------------------------------------------------------
# 对话
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    """智能客服对话请求。"""

    message: str = Field(
        ...,
        description="用户输入的自然语言消息",
        min_length=1,
        max_length=4000,
    )
    session_id: str | None = Field(
        default=None,
        description="会话 ID，用于多轮记忆。不传则每次为新会话",
        max_length=128,
    )
    mode: ChatMode = Field(
        default="auto",
        description=(
            "auto：由 Agent 自主决策调用哪个工具（默认，推荐）；"
            "rag：强制只走政策检索问答，不建工单；"
            "agent：显式走 Agent 编排"
        ),
    )
    top_k: int = Field(
        default=0,
        description="RAG 模式下返回的片段数，0 表示使用系统默认值",
        ge=0,
        le=20,
    )


class SourceRef(BaseModel):
    """回答所依据的知识库片段。"""

    chunk_id: str = ""
    source: str = ""
    section: str = ""
    doc_type: str = ""
    score: float | None = None
    snippet: str = ""


class ToolStep(BaseModel):
    """Agent 执行过的工具调用。"""

    tool: str
    label: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """对话响应。"""

    code: int = 0
    msg: str = "success"
    answer: str = ""
    mode: str = "agent"
    session_id: str = ""
    # RAG 模式：是否基于知识库可答；Agent 模式：是否正常完成
    answerable: bool = True
    sources: list[SourceRef] = Field(default_factory=list)
    steps: list[ToolStep] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    reason: str = ""


# ----------------------------------------------------------------------
# 知识库入库
# ----------------------------------------------------------------------
class IngestRequest(BaseModel):
    """知识库入库请求。"""

    rebuild: bool = Field(
        default=False,
        description="是否清空并全量重建 Milvus 集合（首次入库或更换向量模型时必须为 true）",
    )
    directory: str | None = Field(
        default=None,
        description="指定要入库的目录（默认 data/policies）",
    )


class IngestResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    files: int = 0
    documents: int = 0
    chunks: int = 0
    written: int = 0
    collection_count: int = 0
    elapsed_seconds: float = 0.0
    details: list[dict[str, Any]] = Field(default_factory=list)


# ----------------------------------------------------------------------
# 健康检查
# ----------------------------------------------------------------------
class ComponentStatus(BaseModel):
    name: str
    healthy: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    components: list[ComponentStatus] = Field(default_factory=list)
