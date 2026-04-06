# AI Butler — 一期技术方案

> 目标：构建一个具备**持续记忆能力**的个人 AI 助手，能跨会话记住用户偏好、习惯和重要事件，像真正了解你的人一样对话。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────┐
│                    用户消息输入                        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              查询路由模块（Query Router）              │
│   规则判断 → 直接反应 / 浅层检索 / 深度检索            │
└──────┬──────────────┬───────────────────────────────┘
       │              │
       ▼              ▼
  [浅层档位]      [深度档位]
  用户画像JSON    向量检索(Qdrant)
  最近N轮对话     BM25关键词检索
                  融合重排打分
       │              │
       └──────┬───────┘
              ▼
┌─────────────────────────────────────────────────────┐
│              Context 组装层                           │
│  程序记忆(System Prompt) + 语义记忆 + 情景记忆 + 工作记忆 │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              LLM 推理（OpenAI 兼容接口）               │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              记忆写入（异步）                          │
│  重要信息 → Embedding → Qdrant                        │
│  工具输出 → 压缩 → 文件存储                           │
└─────────────────────────────────────────────────────┘
```

---

## 二、外部依赖

| 组件 | 选型 | 说明 |
|------|------|------|
| **LLM** | 任意兼容 OpenAI 接口的模型（≥ GLM-4 级别） | 推荐智谱 GLM-4-Flash（免费额度大）或 DeepSeek-V3 |
| **Embedding** | 任意兼容 OpenAI 接口的远程 embedding 服务 | 推荐智谱 embedding-3（免费）或 Jina embeddings |
| **向量数据库** | Qdrant（Docker 本地部署） | 支持 payload filter + 时间戳，适合记忆衰减打分 |
| **BM25** | rank_bm25（Python 库） | 纯本地内存运行，无外部依赖 |
| **框架底座** | ReMe（agentscope-ai/ReMe）| 文件式记忆管理，提供 pre/post reasoning hook |
| **对话流程** | LangGraph | 状态机清晰，方便后续加梦境模块等复杂流程 |
| **关系数据库** | ❌ 一期不引入 | 记忆元数据存 Qdrant payload，用户画像存 JSON 文件 |

### 所需配置项（环境变量）

```env
# LLM
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 或任意 OpenAI 兼容地址
LLM_API_KEY=your_api_key
LLM_MODEL=glm-4-flash

# Embedding
EMB_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMB_API_KEY=your_api_key
EMB_MODEL=embedding-3

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=ai_butler_memory

# 记忆参数
MEMORY_TOP_K=10           # 检索返回条数
MEMORY_DECAY_LAMBDA=0.01  # 时间衰减系数
WORKING_MEMORY_TURNS=6    # 工作记忆保留轮数
```

---

## 三、记忆分层设计

```
┌──────────────────────────────────────────────────────┐
│  程序记忆（Procedural Memory）                        │
│  → System Prompt 常驻，人格/行为模式，基本不变         │
│  → 存储：prompt 文件                                  │
├──────────────────────────────────────────────────────┤
│  语义记忆（Semantic Memory）                          │
│  → 用户偏好、习惯、人物关系、长期事实                  │
│  → 存储：user_profile.json（结构化）                  │
├──────────────────────────────────────────────────────┤
│  情景记忆（Episodic Memory）                          │
│  → 具体事件 + 时间戳，可检索                          │
│  → 存储：Qdrant（向量 + metadata）                    │
├──────────────────────────────────────────────────────┤
│  工作记忆（Working Memory）                           │
│  → 当前对话滚动窗口，最近 N 轮                        │
│  → 存储：内存（LangGraph state）                      │
└──────────────────────────────────────────────────────┘
```

### Qdrant 中记忆条目的数据结构

```json
{
  "id": "uuid",
  "vector": [0.1, 0.2, ...],
  "payload": {
    "content": "用户提到他不喜欢吃香菜",
    "context": "讨论午餐时用户说的",
    "memory_type": "episodic",
    "importance": 0.7,
    "created_at": 1712345678,
    "last_accessed_at": 1712345678,
    "access_count": 1,
    "tags": ["饮食偏好", "用户习惯"]
  }
}
```

---

## 四、查询路由模块

一期用**规则判断**，不上分类模型。

### 三档路由逻辑

| 档位 | 触发条件 | Context 构成 | 预期延迟 |
|------|---------|------------|---------|
| 🟢 **直接反应** | 指令类（"帮我算一下"）、打招呼、简单闲聊 | System Prompt + 最近 3 轮 | < 100ms |
| 🟡 **浅层检索** | 涉及用户偏好、近期事件、普通问答 | + user_profile.json + 最近 24h 摘要 | 200~400ms |
| 🔴 **深度检索** | 含"上次"/"以前"/"记得吗"、人名、情感事件 | + Qdrant 向量检索 + BM25，融合 top-10 | < 2s |

### 触发关键词（规则）

```python
DEEP_TRIGGERS = [
    "上次", "之前", "以前", "记得吗", "你还记得",
    "我说过", "我提过", "上回", "那时候"
]

DIRECT_TRIGGERS = [
    "帮我算", "翻译", "今天天气", "几点", "定个闹钟"
]
```

### 动态升级

- 浅层检索 LLM 置信度低（输出包含"我不确定你之前..."）→ 自动升级深度
- 用户追问"更早的呢"→ 强制深度检索

---

## 五、记忆检索与打分

### 检索流程

```
输入 Query
  ├── Embedding → Qdrant 向量检索 → top-20 候选
  └── BM25 分词检索 → top-20 候选
          ↓
       合并去重
          ↓
       融合打分
          ↓
       返回 top-K
```

### 融合打分公式

```
score = α × semantic_score
      + β × bm25_score
      + γ × recency_score
      + δ × importance_score

其中：
  recency_score   = exp(-λ × Δt_days)          # 时间衰减，λ=0.01
  importance_score = base_importance            # 每次被检索命中，+0.05（Hebbian）
  α=0.4, β=0.3, γ=0.2, δ=0.1                  # 一期默认权重，后续可调
```

### Contextual Retrieval（存储时加上下文）

写入 Qdrant 前，先用 LLM 给记忆片段生成一句上下文描述，连同原文一起 embed，提升召回率（参考 Anthropic 的做法，召回率提升约 67%）：

```
原文: "用户说不喜欢吃香菜"
上下文描述: "这是用户在2026年4月讨论午餐时提到的饮食偏好"
Embed 内容: "[上下文描述]\n[原文]"
```

---

## 六、Context 组装

### Token 预算分配（以 8K context 为例）

```
System Prompt（程序记忆）     ≤ 800  tokens
用户画像摘要（语义记忆）       ≤ 500  tokens
检索记忆片段（情景记忆）       ≤ 1500 tokens
工作记忆（最近 6 轮对话）      ≤ 2000 tokens
当前用户输入                  ≤ 500  tokens
LLM 回复预留                 ≤ 2700 tokens
───────────────────────────────────────────
合计                          ≤ 8000 tokens
```

### 工具输出压缩

凡是工具返回值超过 200 tokens 的，压缩后存文件，Context 里只放摘要 + 引用路径。防止工具返回撑爆 context。

---

## 七、项目目录结构

```
AI_Butler/
├── README.md
├── docs/
│   └── tech_spec.md           ← 本文档
├── src/
│   ├── main.py                ← 入口，启动对话循环
│   ├── config.py              ← 从环境变量读取配置
│   ├── router/
│   │   └── query_router.py    ← 查询路由（三档规则）
│   ├── memory/
│   │   ├── episodic.py        ← 情景记忆：Qdrant 读写
│   │   ├── semantic.py        ← 语义记忆：user_profile.json
│   │   ├── working.py         ← 工作记忆：滚动窗口
│   │   ├── retriever.py       ← 检索器：向量 + BM25 + 融合打分
│   │   └── writer.py          ← 记忆写入：异步，含 contextual retrieval
│   ├── context/
│   │   └── assembler.py       ← Context 组装，token 预算管理
│   ├── llm/
│   │   ├── client.py          ← OpenAI 兼容客户端封装
│   │   └── embedding.py       ← Embedding 封装
│   └── tools/
│       └── compressor.py      ← 工具输出压缩
├── data/
│   ├── user_profile.json      ← 用户画像（语义记忆）
│   └── memory_files/          ← 压缩后的工具输出存储
├── docker-compose.yml         ← Qdrant 一键启动
├── requirements.txt
└── .env.example
```

---

## 八、部署方式

### 1. 启动 Qdrant

```bash
docker-compose up -d qdrant
```

`docker-compose.yml`:
```yaml
version: "3"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：
```
openai>=1.0.0
qdrant-client>=1.7.0
rank_bm25>=0.2.2
langgraph>=0.1.0
agentscope[remelight]
tiktoken
python-dotenv
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 填入 LLM_API_KEY、EMB_API_KEY 等
```

### 4. 运行

```bash
python src/main.py
```

---

## 九、一期不做的事（明确边界）

| 功能 | 推迟到 | 原因 |
|------|--------|------|
| 梦境整理模块（异步重写记忆） | 二期 | 一期先积累数据，验证检索效果 |
| 矛盾检测 / 前摄干扰处理 | 二期 | 需要梦境模块支撑 |
| Graphiti 时序知识图谱 | 二期 | 依赖 Neo4j，增加部署复杂度 |
| 查询路由升级为分类模型 | 三期 | 先跑规则，积累数据后再训练 |
| 预判预取（打字时提前检索） | 三期 | 优化项，一期不是瓶颈 |
| MySQL 持久化 | 待评估 | Qdrant payload 够用，再观察 |

---

## 十、一期里程碑

- [ ] **M1**：Qdrant 初始化 + Embedding 写入读取跑通
- [ ] **M2**：BM25 检索 + 向量检索 + 融合打分跑通
- [ ] **M3**：查询路由三档规则实现
- [ ] **M4**：Context 组装 + Token 预算管理
- [ ] **M5**：端到端对话跑通（有记忆的完整对话）
- [ ] **M6**：工具输出压缩实现
- [ ] **M7**：user_profile.json 自动更新（用户偏好提取）

---

*文档版本：v0.1 — 2026-04-06*  
*作者：AI Butler 项目组*
