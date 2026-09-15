# 拾汇商城 AI 智能客服服务

基于 **FastAPI + LangChain + Milvus** 的智能客服，作为独立 Python 微服务与现有
Spring Cloud Alibaba 商城（Nacos 注册、Spring Cloud Gateway 网关、JWT 认证）集成。

---

## 一、能力概览

| 能力 | 实现 |
| --- | --- |
| **RAG 政策问答** | 政策文档（PDF / Word / Markdown / TXT）→ 分块 → 向量化入库 → 检索增强生成；回答**严格基于检索上下文**，检索不到依据时**明确拒答** |
| **混合检索** | LangChain `BM25Retriever`（jieba 中文分词）+ Milvus 向量语义检索，用 `EnsembleRetriever` 做 **RRF 加权倒数排名融合** |
| **Agent 自主决策** | LangChain `create_agent`（ReAct 工具循环），由模型自行决定调用「政策检索」还是「创建投诉工单」 |
| **工单创建工具** | 封装为 LangChain Tool，经 **Spring Cloud Gateway** 调用 Java 后端 `ComplaintController` |
| **JWT 透传** | Python **不解析、不校验** JWT；从 FastAPI 请求头提取后经 `RunnableConfig` / `ToolRuntime` 透传，工具调用网关时原样放入 `Authorization: Bearer <JWT>` |

---

## 二、架构与调用链

```
┌──────────┐   Authorization: Bearer <JWT>
│  前端/APP │───────────────────────────────┐
└──────────┘                               │
                                           ▼
                            ┌──────────────────────────────┐
                            │  Spring Cloud Gateway  :88   │
                            │  /api/ai/**     → AI 服务     │
                            │  /api/member/** → tenhub-member│
                            └──────────────┬───────────────┘
                                           │ lb://tenhub-ai-service
                                           ▼
                                  ┌──────────────────┐        ┌──────────────┐
                                  │  AI 客服服务      │◀──────▶│ Nacos :8848  │
                                  │  FastAPI :8090   │ 注册/发现│ (public ns) │
                                  │                  │        └──────────────┘
                                  │  1. 提取 JWT      │  只提取，不解析
                                  │  2. build_run_config
                                  │  3. Agent 决策    │
                                  └───┬──────────┬───┘
                    政策类问题        │          │      投诉类诉求
                                      ▼          ▼
                        ┌──────────────────┐  ┌────────────────────┐
                        │ 混合检索          │  │ 工单创建 Tool       │
                        │ BM25 + 向量 + RRF │  │ (携带 JWT)          │
                        └────────┬─────────┘  └─────────┬──────────┘
                                 ▼                      │
                        ┌──────────────────┐            │
                        │ Milvus :19530    │            │
                        │ (WSL Docker)     │            │
                        └──────────────────┘            │
                                                        ▼
                                        ┌───────────────────────────────┐
                                        │ 再经网关 :88                   │
                                        │ /api/member/complaint/create  │
                                        └───────────────┬───────────────┘
                                                        ▼
                                        ┌───────────────────────────────┐
                                        │ tenhub-member                 │
                                        │ JwtInterceptor 校验 JWT       │
                                        │ → userId → ComplaintController│
                                        │ → ums_complaint 落库          │
                                        └───────────────────────────────┘
```

### 关键设计：JWT 为什么这样传

1. **Python 侧永不解析 JWT**。签名密钥只存在于 Java/Nacos 配置，Python 服务即使被
   攻破也无法伪造身份。
2. **JWT 不作为工具入参**。工具签名中的 `runtime: ToolRuntime` 由 LangGraph 注入，
   实测**不会出现在模型可见的 tool schema 中**
   （见 `tests/test_agent_tool_passthrough.py`），因此大模型既看不到也无法篡改凭据。
3. **工单归属只认 JWT**。`ComplaintCreateDto` 刻意不含 `userId`，Java 侧一律以
   JWT 的 `subject` 作为投诉人，从根上杜绝「让 AI 替他人建单」的越权。

---

## 三、目录结构

```
ai-customer-service/
├── app/
│   ├── main.py                     # FastAPI 入口 + 启动预热
│   ├── api/
│   │   ├── routes.py               # 路由：对话/流式/检索/入库/健康检查
│   │   └── schemas.py              # 请求响应契约
│   ├── core/
│   │   ├── config.py               # 全部配置项（AI_CS_ 前缀）
│   │   ├── request_context.py      # JWT 经 RunnableConfig/ToolRuntime 透传
│   │   └── logging_config.py
│   ├── rag/
│   │   ├── loader.py               # PDF/Word/MD/TXT 解析 + 章节感知分块
│   │   ├── embeddings.py           # 本地 BGE 向量化（查询侧加指令前缀）
│   │   ├── vectorstore.py          # Milvus 显式 schema 读写
│   │   ├── hybrid_retriever.py     # BM25 + 向量 + RRF 融合
│   │   ├── ingest.py               # 入库编排（Milvus 与 BM25 同源同步）
│   │   └── rag_chain.py            # RAG 三道防线 + 拒答策略
│   ├── agent/
│   │   ├── llm.py                  # DeepSeek（OpenAI 兼容）客户端
│   │   └── customer_service_agent.py # Agent 编排与自主决策
│   ├── tools/
│   │   ├── policy_tools.py         # 政策检索 Tool
│   │   └── complaint_tools.py      # 工单创建/查询 Tool（透传 JWT）
│   └── utils/
│       ├── jwt.py                  # Bearer 提取（不校验）
│       └── tokenizer.py            # jieba 中文分词（BM25 用）
├── data/policies/                  # 政策文档 11 份（用户协议与账户、订单与交易、支付、发票、配送与签收、
│                                   #   运费、退换货、退款、价格保护、会员与优惠券、购物常见问题 FAQ）
├── data/store/                     # BM25 语料快照（自动生成）
├── scripts/
│   ├── download_model.py           # 下载本地向量模型（支持国内镜像）
│   ├── ingest_policies.py          # 命令行入库 + 一致性校验
│   └── make_test_jwt.py            # 联调用 JWT 生成（仅测试，不硬编码密钥）
├── tests/
│   ├── test_units.py               # 24 个纯函数单测
│   ├── test_integration.py         # 29 项集成检查
│   ├── test_tool_contract.py       # 15 项工单工具 HTTP 契约检查
│   ├── test_agent_tool_passthrough.py  # Agent→Tool JWT 透传验证
│   └── test_e2e_live.py            # 10 项真实 Java 后端端到端检查
├── requirements.txt
└── .env.example
```

---

## 四、快速开始

### 1. 前置条件

| 组件 | 要求 | 本项目实测 |
| --- | --- | --- |
| Python | 3.11+ | 3.14.6 |
| Milvus | 2.4+ | v3.0.0（WSL Docker，`127.0.0.1:19530`） |
| Nacos | 已启动 | `127.0.0.1:8848`（AI 服务注册于此，见第十一节） |
| MySQL | 已启动 | `127.0.0.1:3306` |
| Spring Cloud Gateway | 已启动 | `:88`（AI 服务经此暴露为 `/api/ai/**`） |
| tenhub-member | 已启动 | `:10000`（经网关 `/api/member/**` 访问） |

### 2. 创建虚拟环境

```powershell
cd E:\javaProject\拾汇商城\ai-customer-service

# 若本机已全局安装 torch / sentence-transformers / langchain，
# 用 --system-site-packages 复用可节省数 GB 下载
python -m venv --system-site-packages .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt `
    -i https://mirrors.aliyun.com/pypi/simple/
```

### 3. 配置环境变量

```powershell
Copy-Item .env.example .env
notepad .env      # 填入 AI_CS_DEEPSEEK_API_KEY
```

### 4. 下载向量模型（首次一次即可，约 95MB）

```powershell
# 国内推荐走 HF 镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\python.exe scripts\download_model.py --source hf-mirror
```

> 若下载失败，也可手动下载模型目录后用本地路径启动：
> `AI_CS_EMBEDDING_MODEL=D:\models\bge-small-zh-v1.5`

### 5. 建库 + 入库政策文档

```powershell
# 建投诉工单表（幂等：表已存在时只补列与索引，可安全重复执行）
mysql -uroot -proot < E:\javaProject\拾汇商城\SQL\tenhub_ums_complaint.sql

# 政策文档入库（首次务必加 --rebuild）
.\.venv\Scripts\python.exe scripts\ingest_policies.py --rebuild
```

### 6. 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8090
```

- 交互式文档：<http://127.0.0.1:8090/docs>
- 健康检查：<http://127.0.0.1:8090/api/ai/health>

> **端口说明**：默认 `8090`。本机 `8080` 已被一个 Java 进程占用
> （实测 `Get-NetTCPConnection -LocalPort 8080` 指向 java.exe），
> 因此未沿用 8080。如有冲突可用 `AI_CS_PORT` 覆盖。

---

## 五、API 接口

所有接口都需要 `Authorization: Bearer <JWT>`（登录 `tenhub-member` 获取）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/ai/chat` | **统一对话入口**，Agent 自主决策（推荐） |
| POST | `/api/ai/chat/stream` | SSE 流式对话（`tool_start` / `token` / `done`） |
| POST | `/api/ai/rag/query` | 强制只走政策 RAG，不触碰工单接口 |
| GET | `/api/ai/rag/search` | 混合检索调试：查看命中片段与相似度 |
| POST | `/api/ai/rag/ingest` | 政策文档目录入库 |
| POST | `/api/ai/rag/upload` | 上传单个文档并入库 |
| GET | `/api/ai/health` | 健康检查（Milvus / 向量模型 / BM25 / LLM / 网关） |
| GET | `/api/ai/config` | 查看生效配置（已脱敏） |

### 调用示例

```powershell
# 1. 登录拿 JWT
$login = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:88/api/member/member/internal/login" `
  -ContentType "application/json" `
  -Body '{"account":"test@example.com","password":"123456"}'
$jwt = $login.token
$headers = @{ Authorization = "Bearer $jwt" }

# 2. 政策问答（应命中知识库并带引用编号）
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8090/api/ai/chat" `
  -Headers $headers -ContentType "application/json" `
  -Body '{"message":"七天无理由退货的时限是多久？运费谁承担？"}'

# 3. 投诉（Agent 应自动调用工单工具，并带上 JWT 建单）
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8090/api/ai/chat" `
  -Headers $headers -ContentType "application/json" `
  -Body '{"message":"我订单 202501010001 收到的手机屏幕是碎的，商家还不给退，我要投诉！"}'

# 4. 知识库外的问题（应明确拒答）
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8090/api/ai/chat" `
  -Headers $headers -ContentType "application/json" `
  -Body '{"message":"怎么用 Python 训练卷积神经网络？"}'
```

---

## 六、核心机制说明

### 6.1 混合检索与 RRF

- **BM25 侧**：LangChain `BM25Retriever` + jieba 搜索引擎模式分词。
  必须传 `preprocess_func`——默认的英文空白分词会把整句中文当成一个 token，
  关键词召回形同失效。
- **向量侧**：Milvus HNSW + COSINE，本地 `bge-small-zh-v1.5`（512 维）。
  BGE 中文模型**查询侧需加指令前缀**，文档侧不加，`embeddings.py` 已区分处理。
- **融合**：`EnsembleRetriever(weights=[0.5, 0.5], c=60, id_key="chunk_id")`，
  即 `score(d) = Σ wᵢ / (rankᵢ(d) + 60)`，按 `chunk_id` 去重并累加分数。

### 6.2 严格拒答的三道防线

1. **前置门控**：所有命中片段的 COSINE 相似度都低于 `AI_CS_RELEVANCE_THRESHOLD`
   （默认 `0.40`）时，**直接返回固定拒答文案，不调用大模型**——省成本且从机制上杜绝幻觉。
   > 阈值标定：实测相关政策问题最高相似度 0.55~0.75，无关问题约 0.33，取 0.40 留出间隔。
2. **Prompt 约束**：系统提示明确「仅依据资料作答」「资料不足必须拒答」
   「不得编造条款编号、时限、金额」。
3. **后置校验**：识别模型输出的拒答短语，统一 `answerable` 标记与引用来源。

### 6.3 增量入库的一致性

BM25 索引必须常驻内存（`BM25Retriever` 构造时需全量语料），因此分块结果会快照到
`data/store/bm25_corpus.json`。入库时**先写 Milvus 再刷新快照**，保证两路检索同源；
`ingest_policies.py` 结束时会校验两者条数是否一致，不一致会给出告警并以退出码 2 结束。

---

## 七、测试

```powershell
cd E:\javaProject\拾汇商城\ai-customer-service

# 1) 单元测试（无需 Milvus / 大模型）
.\.venv\Scripts\python.exe tests\test_units.py

# 2) Agent→Tool JWT 透传验证（无需大模型，用脚本化假模型走真实工具执行链路）
.\.venv\Scripts\python.exe tests\test_agent_tool_passthrough.py

# 3) 工单工具 HTTP 契约（本地假网关，校验路径/请求头/请求体字段/错误分支）
.\.venv\Scripts\python.exe tests\test_tool_contract.py

# 4) 集成测试（需 Milvus + 向量模型）
.\.venv\Scripts\python.exe tests\test_integration.py
.\.venv\Scripts\python.exe tests\test_integration.py --skip-ingest   # 已入库时

# 5) 端到端（需 tenhub-member 正在运行 + MySQL 已建表）
.\.venv\Scripts\python.exe tests\test_e2e_live.py --from-nacos

# 6) 经网关验证 Nacos 发现与路由（需 gateway + AI 服务都在运行）
$jwt = .\.venv\Scripts\python.exe scripts\make_test_jwt.py --user-id 1 --from-nacos
.\.venv\Scripts\python.exe scripts\verify_gateway_ai.py --jwt $jwt
```

### 实测结果（全部通过）

| 测试 | 规模 | 结果 |
| --- | --- | --- |
| `test_units.py` | 24 个单测 | **24/24 通过** |
| `test_agent_tool_passthrough.py` | 4 项（含 schema 不外泄 runtime） | **全部通过** |
| `test_tool_contract.py` | 15 项契约检查 | **15/15 通过** |
| `test_integration.py` | 29 项集成检查 | **29/29 通过** |
| `test_e2e_live.py` | 10 项（真实 Java 后端） | **10/10 通过** |

`test_e2e_live.py` 的实际验证内容：

1. Python 工单工具向真实 `tenhub-member` 建单成功，拿到工单号；
2. 用同一 JWT 查询能查到该工单 → 证明落库的 `user_id` **确实取自 JWT**；
3. **伪造签名的 JWT 被后端拒绝（401）**，工具如实报错、未谎报成功；
4. 缺少 JWT 时工具层直接拒绝，**未向后端发出任何请求**；
5. 返回体不含 `userId`，联系电话已脱敏为 `139****9000`。

> 补充：后续已把 Gateway(`:88`)**真正启动**并完成经网关的完整验证 ——
> 包括网关经 Nacos 发现 AI 服务、经网关调 AI 接口、以及经网关建工单，
> 全部通过（见第十一节 11.4）。E2E 脚本仍保留直连 member 的模式，
> 便于在不启动网关时单独验证 Java 侧。
> 所有测试产生的工单数据均已清理。

---

## 八、Java 侧配套改动

| 文件 | 说明 |
| --- | --- |
| `ComplaintController.java` | 新增。`/member/complaint/create`、`/list`、`/info/{id}`、`/cancel/{id}`，全部 `@RequireLogin` |
| `ComplaintService.java` / `ComplaintServiceImpl.java` | 新增。创建与按用户分页查询 |
| `ComplaintStatusEnum.java` | 新增。**状态取值对齐数据库已有注释**（0-正在受理 / 1-已返回结果待用户确认 / 2-已办结） |
| `ComplaintCreateDto.java` | 新增。**刻意不含 userId**，身份只来自 JWT |
| `ComplaintVo.java` | 新增。返回前端/AI 时不暴露 userId，手机号脱敏 |
| `MaskUtil.java` | 新增。个人信息脱敏 |
| `ComplaintEntity.java` | 补充 `complaintType`、`contactPhone` 字段，并校正 status 注释 |
| `MyBatisConfig.java` | **修复**：补上缺失的 `@Configuration`。原文件没有该注解，`@Bean` 未被注册，分页插件实际失效，所有 `page()` 查询会退化为全表查询 |
| `tenhub-gateway/.../application.yml` | **新增路由** `ai_service_route`（`lb://tenhub-ai-service`，`/api/ai/**`，`order: 1`），使 AI 服务经网关统一暴露 |
| `SQL/tenhub_ums_complaint.sql` | 新增（幂等）。表不存在则建表；已存在则补 `complaint_type`/`contact_phone` 两列与索引 |

### 8.1 关于 ums_complaint 的现状（重要）

排查时发现数据库里 **`ums_complaint` 表已经存在**，且状态语义与新建表不同：

```sql
status tinyint DEFAULT 0 COMMENT '受理状态（0-正在受理，1-已返回结果待用户确认，2-已办结）'
```

因此本项目**没有另起炉灶定义 5 个状态**，而是让 `ComplaintStatusEnum` 严格对齐这
3 个已有语义，避免出现「数据库里是 2（已办结）、代码却当成 4（已关闭）」的错位。
仅新增了两个可空列（`complaint_type`、`contact_phone`），用幂等 `ALTER` 补齐，
不重建、不迁移数据，可直接重复执行。

Python 侧的 `_STATUS_TEXT` 映射也已同步为同一套中文文案。

> ⚠️ `MyBatisConfig` 的修复会影响 `tenhub-member` 现有分页接口的行为（从「返回全部」
> 变为「正确分页」），前端若有依赖旧行为的逻辑需一并确认。

---

## 九、配置项速查

完整清单见 `.env.example`。最常调整的：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `AI_CS_DEEPSEEK_API_KEY` | 空 | **必填**，DeepSeek 控制台申请 |
| `AI_CS_CHAT_MODEL` | `deepseek-chat` | 对话模型 |
| `AI_CS_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 向量模型（本地路径亦可） |
| `AI_CS_MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 地址 |
| `AI_CS_CHUNK_SIZE` / `_OVERLAP` | `500` / `80` | 分块参数 |
| `AI_CS_DENSE_TOP_K` / `_SPARSE_TOP_K` | `8` / `8` | 两路召回条数 |
| `AI_CS_RRF_C` | `60` | RRF 常数 |
| `AI_CS_RELEVANCE_THRESHOLD` | `0.40` | 相关性门控阈值，调高更保守 |
| `AI_CS_GATEWAY_BASE_URL` | `http://127.0.0.1:88` | 网关地址（**不直连 member**） |

---

## 十、常见问题

**Q：启动时报「未配置 AI_CS_DEEPSEEK_API_KEY」？**
`/health` 与 `/rag/*` 接口仍可用，但 `/chat` 与 `/rag/query` 需要大模型。
请确认 `.env` 存在且已填 Key。

**Q：`/rag/search` 返回结果但 `/chat` 总是拒答？**
说明召回片段相似度低于 `AI_CS_RELEVANCE_THRESHOLD`。用 `/rag/search` 看
`score` 与 `passed_threshold` 字段，必要时下调阈值或补充政策文档。

**Q：改政策文档后检索不到新内容？**
需重新入库：`scripts\ingest_policies.py --rebuild`，或调用
`POST /api/ai/rag/ingest`。

**Q：换向量模型后报维度不匹配？**
Milvus 集合维度在创建时固定。换模型必须 `--rebuild` 重建集合，
并同步修改 `AI_CS_EMBEDDING_DIM`。

**Q：多副本部署后多轮对话记忆错乱？**
`InMemorySaver` 仅限单进程。多副本需把 checkpointer 换成 Redis/Postgres 实现
（`app/agent/customer_service_agent.py` 中已标注替换位置）。

**Q：想让网关也暴露 AI 服务（`:88/api/ai/**`）？**
见下一节。

---

## 十一、Nacos 服务注册（已接入）

AI 服务**已注册到 Nacos**，网关可通过 `lb://tenhub-ai-service` 发现并路由，
与其它 Java 微服务一致。前端统一走网关 `:88`：

```
前端 -> http://127.0.0.1:88/api/ai/**  ->  网关 -> lb://tenhub-ai-service -> AI 服务 :8090
```

### 11.1 实测接入参数（照抄即可）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 注册地址 | `127.0.0.1:8848` | Nacos 服务地址 |
| **命名空间** | **public（留空）** | 见下方说明，**不要**填 gateway 配置里的 `d5360b4f...` |
| 分组 | `DEFAULT_GROUP` | |
| 服务名 | `tenhub-ai-service` | |
| **注册 IP** | **`172.29.94.120`** | **必须显式指定**，见下方说明 |
| 注册端口 | `8090` | |

**为什么命名空间用 public？**
gateway 的 `bootstrap.yml` 里写的是 `namespace: d5360b4f-4474-4d40-a1a5-6c59978485d9`，
但那是 `spring.cloud.nacos.config.namespace`，**只作用于配置中心**。实测
`tenhub-member` 注册在 **public**，且网关能正常路由到它（返回 200），
说明网关的**服务发现**用的是 public。所以 AI 服务也必须注册到 public。

**为什么必须显式指定 IP？**
本机有 5 张网卡（WSL `172.17.96.1`、Hyper-V `172.19.224.1`/`192.168.32.1`、
WLAN `172.29.94.120` 等）。Nacos SDK 自动探测很可能选中虚拟网卡地址，
网关按该地址回调会连接失败。实测 `tenhub-member` 注册的 IP 是 `172.29.94.120`，
AI 服务必须用同一个。

> 换机器/换网络后请重新确认：`Get-NetIPAddress -AddressFamily IPv4`

### 11.2 配置方式

在 `.env` 中设置（`.env.example` 已给出带注释的模板）：

```ini
AI_CS_NACOS_ENABLED=true
AI_CS_NACOS_SERVER_ADDR=127.0.0.1:8848
AI_CS_NACOS_NAMESPACE=
AI_CS_NACOS_GROUP=DEFAULT_GROUP
AI_CS_NACOS_SERVICE_NAME=tenhub-ai-service
AI_CS_NACOS_IP=172.29.94.120
```

### 11.3 网关路由（已添加）

`TenHub/tenhub-gateway/src/main/resources/application.yml` 中新增：

```yaml
        - id: ai_service_route
          uri: lb://tenhub-ai-service
          predicates:
            - Path=/api/ai/**
          order: 1
```

> `order: 1` 是必需的：兜底的 `renren_fast_route` 匹配 `/api/**`（`order: 2`），
> 若 AI 路由不加更高优先级，`/api/ai/**` 会被 renren-fast 抢走。
> 另外无需 `RewritePath`，因为 AI 服务自身的路由前缀就是 `/api/ai/**`。

### 11.4 验证结果（实测）

| 检查 | 结果 |
| --- | --- |
| Nacos 中出现 `tenhub-ai-service` 实例 | ✅ `172.29.94.120:8090`，healthy |
| 经网关 `GET /api/ai/health` | ✅ HTTP 200，`nacos` 组件 healthy |
| 经网关 `GET /api/ai/config`（确认未被 renren-fast 抢走） | ✅ 返回 AI 服务配置 |
| 经网关 `GET /api/ai/rag/search` 混合检索 | ✅ 命中退换货政策 0.6315 |
| 经网关 `POST /api/member/complaint/create` | ✅ 工单落库，JWT 透传正常 |
| 经网关无 JWT 访问 `/api/ai/chat` | ✅ 401 |
| 服务关闭后自动注销 | ✅ Nacos 实例消失，网关返回 503 |

验证脚本：`scripts/verify_gateway_ai.py`（用法见脚本头部说明）。

### 11.5 SDK 版本适配说明

`nacos-sdk-python` 3.x 的 API 与 2.x **完全不同**，容易踩坑，已在
`app/core/nacos_registry.py` 中做了适配：

* **3.x**：`from v2.nacos import NacosNamingService, ClientConfig, ...`（注意导入名是
  `v2.nacos` 而不是 `nacos`），且**全部是异步方法**；
* **2.x 及更早**：`import nacos; nacos.NacosClient(...)`，同步方法。

代码会优先尝试 3.x，失败则回退 2.x。另外 3.x 的 `ClientConfigBuilder` 方法名是
`namespace_id()` 而非 `namespace()`，为避免这类易变 API，代码直接构造
`ClientConfig` 数据类。
