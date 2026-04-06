# AI Butler — 一期技术方案 v0.2

> 目标：构建一个具备**持续记忆能力**的个人 AI 助手，能跨会话记住用户偏好、习惯和重要事件，像真正了解你的人一样对话。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户消息输入                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   查询路由（Query Router）                    │
│         规则判断 → 直接反应 / 浅层检索 / 深度检索              │
└──────────┬────────────────┬────────────────────────────────-┘
           │                │
           ▼                ▼
      [浅层档位]         [深度档位]
      用户画像文件        向量检索（Qdrant）
      短期记忆摘要        BM25 关键词检索
                          融合重排打分
           │                │
           └────────┬───────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Context 组装层                          │
│   程序记忆（System Prompt，固定）                            │
│ + 语义记忆（user_profile，每次加载）                         │
│ + 情景记忆（检索结果，按 token 预算截断）                     │
│ + 短期记忆（当前会话历史，含压缩摘要段）                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                LLM 推理（OpenAI 兼容接口）                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    回复后异步处理                             │
│  ① 短期记忆追加落盘（append-only 原文日志）                   │
│  ② 判断是否触发短期记忆压缩                                   │
│  ③ 重要信息 → Embedding → Qdrant + 长期记忆文件              │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、外部依赖

| 组件 | 选型 | 说明 |
|------|------|------|
| **LLM** | 兼容 OpenAI 接口（≥ GLM-5 级别） | 当前：百度千帆 glm-5，base_url 可配置 |
| **Embedding** | 兼容 OpenAI 接口的远程服务 | 当前：OpenRouter nvidia/llama-nemotron-embed-vl-1b-v2:free，2048 维 |
| **向量数据库** | Qdrant（Docker 本地部署） | 长期记忆语义索引，支持 payload filter |
| **BM25** | rank_bm25（Python 库） | 内存索引，启动时从长期记忆文件重建 |
| **关系数据库** | ❌ 一期不引入 | 用文件系统 + Qdrant 覆盖所有存储需求 |

### 环境变量（.env，不进 git）

```env
LLM_BASE_URL=https://qianfan.baidubce.com/v2/coding
LLM_API_KEY=...
LLM_MODEL=glm-5

EMB_BASE_URL=https://openrouter.ai/api/v1
EMB_API_KEY=...
EMB_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=ai_butler_memory

# 短期记忆
SHORT_TERM_SOFT_LIMIT=50000    # token 数超过此值触发压缩
SHORT_TERM_HARD_LIMIT=150000   # 绝对上限，模型 context window 内

# 长期记忆检索
MEMORY_TOP_K=10
MEMORY_DECAY_LAMBDA=0.01
```

---

## 三、记忆分层设计

```
┌─────────────────────────────────────────────────────────────┐
│  程序记忆（Procedural Memory）                               │
│  → 人格、行为规范，写死在 System Prompt 里                   │
│  → 存储：src/prompts/system.txt                              │
│  → 每次对话固定加载，不参与检索                               │
├─────────────────────────────────────────────────────────────┤
│  语义记忆（Semantic Memory）                                 │
│  → 用户偏好、习惯、人物关系等结构化事实                       │
│  → 存储：data/user_profile.json                              │
│  → 每次对话开始时全量加载进 Context                          │
│  → 由 LLM 在对话中自动识别并更新                             │
├─────────────────────────────────────────────────────────────┤
│  情景记忆（Episodic Memory）—— 长期记忆                      │
│  → 具体事件、对话摘要片段，带时间戳                          │
│  → 索引：Qdrant（向量检索）+ BM25（关键词检索）              │
│  → 原文：data/memory_store/YYYY-MM.jsonl（按月分文件）       │
│  → 按需检索，不全量加载                                      │
├─────────────────────────────────────────────────────────────┤
│  短期记忆（Working Memory）—— 当前会话                       │
│  → 当前会话的完整对话历史（原始轮次 + 压缩摘要段混合）        │
│  → 存储：内存（运行时）+ data/sessions/<session_id>.jsonl    │
│  → 全量落盘，进程异常重启时可恢复                            │
└─────────────────────────────────────────────────────────────┘
```

### 长期记忆存储结构

**Qdrant** 存索引（向量 + metadata）：
```json
{
  "id": "uuid",
  "vector": [/* 2048维 */],
  "payload": {
    "content": "用户提到他不喜欢吃香菜",
    "context_desc": "2026年4月，讨论午餐时用户提到的饮食偏好",
    "importance": 0.7,
    "created_at": 1712345678,
    "last_accessed_at": 1712345678,
    "access_count": 1,
    "source_file": "data/memory_store/2026-04.jsonl",
    "source_line": 42
  }
}
```

**文件系统** 存原文（source of truth，Qdrant 只是索引）：
```
data/memory_store/
  2026-04.jsonl   ← 每行一条 JSON，append-only，永不修改
  2026-05.jsonl
  ...
```

---

## 四、短期记忆管理（核心）

### 结构

短期记忆不是纯粹的对话轮次列表，而是**段落混合结构**：

```
[摘要段A]    ← 被压缩过的历史，LLM 生成的结构化摘要
[摘要段B]    ← 可能有多个摘要段（多次压缩）
[原始轮次]   ← 最近未压缩的对话，保持原文
[原始轮次]
[原始轮次]
```

### 压缩触发条件

```
短期记忆 token 数 > SHORT_TERM_SOFT_LIMIT（50k）
  → 触发部分压缩
```

### 压缩流程

```
1. 取出最旧的一批原始轮次（直到 token 数减少到 ~30k）
2. 用 LLM 压缩这批轮次 → 生成结构化摘要段
3. 将摘要段替换掉原始轮次，插入到短期记忆头部
4. 同时：从这批轮次中提取重要事件 → 写入长期记忆（Qdrant + 文件）
5. 更新落盘文件
```

压缩后的摘要格式（便于 LLM 理解）：
```
[对话摘要 2026-04-06 18:00~20:00]
- 用户讨论了 AI Butler 项目的技术方案
- 确定了 Qdrant + BM25 混合检索方案
- LLM 选型：百度千帆 glm-5；Embedding：OpenRouter nvidia
- 用户偏好直接落地，不喜欢纠结细节参数
```

### 落盘策略

- 每轮对话结束后，**立即 append** 到 `data/sessions/<session_id>.jsonl`
- 文件格式：每行一个 JSON，包含 role / content / timestamp / is_summary
- 进程重启时，从文件恢复短期记忆（加载最后一个 session 文件）

---

## 五、长期记忆写入

### 触发时机

- 短期记忆压缩时：从被压缩的轮次中提取重要事件写入
- 每轮对话后：对当前这一轮做轻量判断，重要性 > 0.5 则写入

### 写入流程

```
1. LLM 评估这段内容的重要性（0~1）和类别（事件/偏好/习惯/关系）
2. 如果是偏好/习惯类 → 更新 user_profile.json（语义记忆）
3. 如果是事件类且重要性 > 0.5：
   a. 用 LLM 生成一句 context_desc（Contextual Retrieval 方案）
   b. 对"[context_desc]\n[content]"做 Embedding
   c. 写入 Qdrant
   d. Append 原文到 data/memory_store/YYYY-MM.jsonl
```

---

## 六、查询路由

三档规则路由，不上分类模型：

| 档位 | 触发条件 | Context 构成 |
|------|---------|------------|
| 🟢 直接反应 | 指令类、简单闲聊、数学计算 | System Prompt + 最近 3 轮 |
| 🟡 浅层检索 | 默认档位 | + user_profile + 短期记忆全部 |
| 🔴 深度检索 | 含回忆类关键词、人名、情感事件 | + 向量检索 + BM25 融合 top-K |

深度检索触发词：`上次、之前、以前、记得吗、你还记得、我说过、我提过、上回、那时候`

动态升级：LLM 回复中出现"我不太确定你之前..."等不确定表述 → 自动升级深度并补充检索。

---

## 七、长期记忆检索（深度档位）

### 混合检索流程

```
Query
  ├── Embedding → Qdrant 向量检索 → top-20
  └── BM25（内存索引，启动时从 memory_store/*.jsonl 重建）→ top-20
          ↓
       合并去重
          ↓
       融合打分
          ↓
       top-K 结果
```

### 融合打分

```
score = 0.4 × semantic_score
      + 0.3 × bm25_score
      + 0.2 × recency_score      # exp(-0.01 × Δt天数)
      + 0.1 × importance_score

命中后：importance += 0.05（Hebbian 强化，回写 Qdrant payload）
```

### BM25 索引生命周期

- 启动时：遍历 `data/memory_store/*.jsonl`，加载所有 content 字段，在内存中构建索引
- 新记忆写入后：增量更新内存索引（直接 append，不重建）
- 重启后：重新从文件全量构建（文件是 source of truth）

---

## 八、Context 组装与 Token 预算

```
System Prompt（固定）              ≤ 800  tokens
user_profile 摘要（每次全量）      ≤ 500  tokens
长期记忆检索结果（深度档位才有）    ≤ 1500 tokens
短期记忆（含摘要段 + 最近原始轮次） ≤ 动态（剩余预算）
当前用户输入                       ≤ 500  tokens
LLM 回复预留                      ≤ 2000 tokens
```

短期记忆的 token 占用 = 总预算 - 其他固定部分。超出时从最旧的段落开始丢弃（已有摘要段保护了信息不丢失）。

---

## 九、数据目录结构

```
data/
  user_profile.json          ← 语义记忆（用户画像）
  sessions/
    <session_id>.jsonl       ← 短期记忆落盘（每个会话一个文件）
  memory_store/
    2026-04.jsonl            ← 长期记忆原文（按月，append-only）
    2026-05.jsonl
  qdrant_storage/            ← Qdrant 数据目录（Docker volume）
```

---

## 十、代码目录结构

```
AI_Butler/
├── docs/
│   └── tech_spec.md
├── src/
│   ├── main.py                  ← CLI 入口，对话主循环
│   ├── config.py                ← 环境变量读取
│   ├── prompts/
│   │   └── system.txt           ← 程序记忆（人格定义）
│   ├── memory/
│   │   ├── short_term.py        ← 短期记忆：结构管理 + 压缩触发
│   │   ├── long_term.py         ← 长期记忆：Qdrant + 文件读写
│   │   ├── retriever.py         ← 混合检索 + 融合打分
│   │   └── writer.py            ← 记忆写入决策（重要性判断 + 分流）
│   ├── router/
│   │   └── query_router.py      ← 三档路由规则
│   ├── context/
│   │   └── assembler.py         ← Context 组装 + token 预算管理
│   └── llm/
│       ├── client.py            ← LLM 调用封装（流式 + 非流式）
│       └── embedding.py         ← Embedding 调用封装
├── data/                        ← 运行时数据（不进 git）
├── docker-compose.yml           ← Qdrant
├── requirements.txt
├── .env                         ← 敏感配置（不进 git）
└── .env.example                 ← 配置模板
```

---

## 十一、一期不做的事

| 功能 | 推迟到 | 原因 |
|------|--------|------|
| 梦境模块（全局重写记忆） | 二期 | 一期只做部分压缩，梦境做全局去重/矛盾检测 |
| 矛盾检测 / 前摄干扰处理 | 二期 | 梦境模块负责 |
| Graphiti 时序图谱 | 二期 | 一期 Qdrant + BM25 够用 |
| 查询路由升级分类模型 | 三期 | 积累数据后再训练 |
| 错误处理 / 降级策略 | 待定 | Qdrant 挂了降级为纯短期记忆模式 |
| 多用户支持 | 待定 | 一期只考虑单用户 |

---

## 十二、一期里程碑

- [ ] **M1**：Qdrant 初始化 + Embedding 写入/读取跑通
- [ ] **M2**：BM25 内存索引构建 + 混合检索 + 融合打分跑通
- [ ] **M3**：短期记忆管理（结构 + 落盘 + 压缩触发）
- [ ] **M4**：长期记忆写入（重要性判断 + Contextual Retrieval）
- [ ] **M5**：查询路由三档规则
- [ ] **M6**：Context 组装 + Token 预算管理
- [ ] **M7**：端到端 CLI 对话跑通（有记忆的完整对话）

---

*v0.2 — 2026-04-06，纳入短期记忆压缩策略、长期记忆两层存储、BM25 索引生命周期、全量落盘方案*
