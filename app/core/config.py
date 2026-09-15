"""应用配置。

所有可调参数集中在此，通过环境变量或 .env 文件覆盖。
命名前缀统一为 ``AI_CS_``，避免污染其它服务的环境变量。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：.../ai-customer-service
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """智能客服服务配置。"""

    model_config = SettingsConfigDict(
        env_prefix="AI_CS_",
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # 服务自身
    # ------------------------------------------------------------------
    app_name: str = "tenhub-ai-customer-service"
    host: str = "0.0.0.0"
    # 注意：8080 在本机已被一个 Java 进程占用，默认改用 8090。
    # 如有冲突可通过 AI_CS_PORT 覆盖。
    port: int = 8090
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # LLM（DeepSeek，OpenAI 兼容协议）
    # ------------------------------------------------------------------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    chat_model: str = "deepseek-chat"
    # 生成温度：政策问答要求稳定、可复现，取低值
    temperature: float = 0.0
    request_timeout: float = 120.0
    max_retries: int = 2

    # ------------------------------------------------------------------
    # Embedding（本地 BGE 中文模型，离线可用）
    # ------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_device: str = "cpu"
    # 向量检索时是否对 query 加 BGE 官方推荐前缀（检索质量提升明显）
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    # 归一化向量后可用内积；保持 True 以配合 COSINE 度量
    embedding_normalize: bool = True

    # ------------------------------------------------------------------
    # Milvus
    # ------------------------------------------------------------------
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_collection: str = "tenhub_policy_chunks"
    milvus_timeout: float = 30.0

    # ------------------------------------------------------------------
    # 检索参数
    # ------------------------------------------------------------------
    chunk_size: int = 500
    chunk_overlap: int = 80
    # 单路召回条数（BM25 / 向量各取 top-k）
    dense_top_k: int = 8
    sparse_top_k: int = 8
    # 融合后返回给 Agent 的条数
    fusion_top_k: int = 6
    # RRF 融合权重：[向量语义, BM25 关键词]，两者之和不必为 1
    rrf_weight_dense: float = 0.5
    rrf_weight_sparse: float = 0.5
    # RRF 常数，论文默认 60
    rrf_c: int = 60
    # 相关性门控阈值：低于该 COSINE 相似度的片段不作为作答依据（触发拒答）。
    # 标定依据（BAAI/bge-small-zh-v1.5，中文）：实测「确有相关政策」的问题
    # 最高相似度在 0.55~0.75；而与政策无关的问题（如技术类提问）最高仅 0.33 左右。
    # 取 0.40 使两者之间留出安全间隔：宁可拒答（可兜底转人工）也不编造政策。
    relevance_threshold: float = 0.40

    # ------------------------------------------------------------------
    # Nacos 服务注册与发现
    #
    # 重要：本机有多个网卡（含 WSL / Hyper-V 虚拟网卡），Nacos SDK 自动探测
    # 出的 IP 很可能是 172.17.x.x 这类虚拟网卡地址，导致 Spring Cloud Gateway
    # 无法回调。因此 **必须显式指定** nacos_ip 为本机局域网 IP，
    # 且要与其它 Java 服务注册用的 IP 一致（实测 tenhub-member 用的是
    # 172.29.94.120）。留空时才退化为自动探测。
    # ------------------------------------------------------------------
    nacos_enabled: bool = True
    nacos_server_addr: str = "127.0.0.1:8848"
    # 命名空间：留空 = public。
    # 实测网关的服务发现用的是 public（gateway 配置里的 namespace
    # d5360b4f... 只作用于配置中心），而 tenhub-member 也注册在 public，
    # 因此本服务同样注册到 public 才能被网关发现。
    nacos_namespace: str = ""
    nacos_group: str = "DEFAULT_GROUP"
    nacos_service_name: str = "tenhub-ai-service"
    # 留空则自动探测（不推荐，见上）
    nacos_ip: str = ""
    # 注册端口，留空则用本服务 port
    nacos_port: int = 0
    # 心跳间隔（秒）
    nacos_heartbeat_interval: float = 5.0
    # Nacos 鉴权（未开启鉴权时留空）
    nacos_username: str = ""
    nacos_password: str = ""

    # ------------------------------------------------------------------
    # 知识库与持久化
    # ------------------------------------------------------------------
    # 政策文档放置目录
    policy_dir: Path = BASE_DIR / "data" / "policies"
    # 本地持久化目录（BM25 语料快照、入库清单）
    store_dir: Path = BASE_DIR / "data" / "store"
    # 支持的文档后缀
    supported_suffixes: tuple[str, ...] = (".md", ".markdown", ".txt", ".pdf", ".docx")

    # ------------------------------------------------------------------
    # Java 后端集成（经 Spring Cloud Gateway）
    # ------------------------------------------------------------------
    # 不直连 tenhub-member，一律经网关，由网关/后端完成 JWT 鉴权
    gateway_base_url: str = "http://127.0.0.1:88"
    # 网关路由前缀：/api/member/** -> tenhub-member
    complaint_create_path: str = "/api/member/complaint/create"
    complaint_list_path: str = "/api/member/complaint/list"
    gateway_timeout: float = 15.0
    # 内部服务标识头，便于网关侧日志与限流区分调用来源
    internal_caller: str = "ai-customer-service"

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    agent_recursion_limit: int = 12
    # 会话记忆保留的最近消息条数
    memory_window: int = 12

    @field_validator("policy_dir", "store_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    @property
    def bm25_corpus_path(self) -> Path:
        """BM25 语料快照文件。

        BM25Retriever 需要在构造时载入全部语料并在内存建索引，
        因此把分块结果落盘，服务启动时直接载入，无需重新解析原始文档。
        """
        return self.store_dir / "bm25_corpus.json"

    @property
    def ingest_manifest_path(self) -> Path:
        """入库清单，记录已索引文件的内容指纹，用于增量判定。"""
        return self.store_dir / "ingest_manifest.json"

    @property
    def llm_configured(self) -> bool:
        return bool(self.deepseek_api_key.strip())

    @property
    def nacos_register_port(self) -> int:
        """实际向 Nacos 注册的端口：未单独指定时用服务监听端口。"""
        return self.nacos_port or self.port


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（带缓存）。"""
    return Settings()


settings = get_settings()
