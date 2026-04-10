# AI Butler — 一期技术方案 v0.3

> 目标：构建一个能像人一样持续记忆、理解用户、自然对话的个人 AI 助手。
> 核心挑战：跨会话记忆管理——分辨相关与不相关信息，能回忆近期和远期记忆，不"失忆"。

---

## 一、工程底座：ReMe

一期以 **ReMe（agentscope-ai/ReMe）** 为工程底座，不从零造轮子。

ReMe 是阿里出品的文件式记忆管理框架，核心是 `pre_reasoning_hook` 四步流程，与我们的需求高度对齐：

```
每轮对话前，按顺序执行：

Step 1: compact_tool_result
  → 压缩上一轮的长工具输出，存文件，Context 里只留摘要+引用
  → 防止工具返回值撑爆 context（家庭管家大量调用工具，这是必须的）

Step 2: check_context
  → 用 tiktoken 计算当前 context token 数
  → 判断是否超过短期记忆阈值（soft limit: 50k tokens）

Step 3: compact_memory（触发条件：超过阈值）
  → 取出最旧的一批原始对话轮次
  → LLM 压缩为结构化摘要段，替换掉原始轮次（部分压缩，非全局重写）
  → 同时将重要事件提取出来，写入长期记忆

Step 4: summary_memory（异步，不阻塞主对话）
  → 对刚压缩的内容做语义提炼
  → 更新用户画像（语义记忆）
  → 将重要片段沉淀到向量数据库（情景记忆）
```

上层对话流程用 **LangGraph** 管理状态机，ReMe hook 作为节点嵌入其中。

---

## 二、外部依赖

| 组件 | 选型 | 说明 |
|------|------|------|
| **工程底座** | ReMe（agentscope-ai/ReMe） | pre_reasoning_hook 四步流程 |
| **对话框架** | LangGraph | 状态机，管理对话主循环 |
| **LLM** | 兼容 OpenAI 接口（≥ GLM-5） | 当前：百度千帆 glm-5 |
| **Embedding** | 兼容 OpenAI 接口的远程服务 | 当前：OpenRouter nvidia/llama-nemotron-embed-vl-1b-v2:free，2048 维 |
| **向量数据库** | Qdrant（Docker） | 情景记忆语义索引 |
| **BM25** | rank_bm25（Python 库） | 内存索引，精确匹配人名/地名/数字 |
| **时序图谱** | ❌ 二期引入 Graphiti | 一期 Qdrant + BM25 覆盖检索需求 |
| **关系数据库** | ❌ 一期不引入 | 文件系统 + Qdrant 覆盖所有存储 |

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

SHORT_TERM_SOFT_LIMIT=50000     # token 数超过此值触发压缩
SHORT_TERM_HARD_LIMIT=150000    # 模型 context window 绝对上限

MEMORY_TOP_K=10
MEMORY_DECAY_LAMBDA=0.01        # 时间衰减系数 λ
```

---

## 三、记忆分层设计

参考认知科学的四类记忆，对应不同存储和使用方式：

```
┌──────────────────────────────────────────────────────────────┐
│  程序记忆（Procedural Memory）                                │
│  → 人格、行为规范、对话风格                                   │
│  → 存储：src/prompts/system.txt                               │
│  → 每次固定加载进 System Prompt，不参与检索                   │
├──────────────────────────────────────────────────────────────┤
│  语义记忆（Semantic Memory）                                  │
│  → 用户偏好、习惯、人物关系等结构化事实                        │
│  → 存储：data/user_profile.json                               │
│  → 每次对话开始全量加载，由梦境模块异步更新                    │
│  → ⚠️ 必须精炼（< 500 tokens），否则挤占推理空间               │
├──────────────────────────────────────────────────────────────┤
│  情景记忆（Episodic Memory）—— 长期记忆                       │
│  → 具体事件、对话摘要片段，带时间戳                           │
│  → 索引层：Qdrant（向量检索）+ BM25 内存索引（关键词检索）     │
│  → 原文层：data/memory_store/YYYY-MM.jsonl（append-only）     │
│         Qdrant 是索引，文件是 source of truth                 │
│  → 按需检索，不全量加载                                       │
├──────────────────────────────────────────────────────────────┤
│  工作记忆（Working Memory）—— 短期记忆                        │
│  → 当前会话完整对话历史                                       │
│  → 结构：[摘要段...] + [原始轮次...]（混合，见下节）           │
│  → 内存运行 + 全量落盘（data/sessions/<id>.jsonl）            │
│  → 上限：soft 50k / hard 150k tokens                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 四、短期记忆（工作记忆）详细设计

### 混合段落结构

短期记忆不是纯粹的轮次列表，而是**摘要段 + 原始轮次**的混合结构：

```
[摘要段 A]  ← 最早的历史，已被压缩（LLM 生成）
[摘要段 B]  ← 第二批历史，已被压缩
[原始轮次]  ← 最近未压缩的对话，保持原文
[原始轮次]
[原始轮次]  ← 当前
```

### 压缩触发（对应 ReMe Step 2 & 3）

```
token 数 > 50k（soft limit）
  → 取出最旧的一批原始轮次（目标：压缩后降至 ~30k）
  → LLM 生成结构化摘要段，替换掉这批原始轮次
  → 被压缩的内容同步提交给梦境模块处理（异步）
```

压缩摘要格式（便于 LLM 理解时序）：
```
[对话摘要 2026-04-06 18:00~20:00]
- 讨论了 AI Butler 项目记忆系统架构
- 确定 Qdrant + BM25 混合检索方案
- LLM 选型：百度千帆 glm-5
- 用户偏好：直接落地，不纠结细节参数
```

### 落盘策略

每轮对话结束后立即 append 到 `data/sessions/<session_id>.jsonl`：
```json
{"role": "user", "content": "...", "timestamp": 1234567890, "is_summary": false}
{"role": "assistant", "content": "...", "timestamp": 1234567891, "is_summary": false}
{"role": "summary", "content": "...", "timestamp": 1234567900, "is_summary": true, "covers_range": [100, 200]}
```
进程崩溃后从文件恢复，不丢对话原文。

---

## 五、梦境模块（Dream Agent）

**梦境是本系统最核心的模块之一**，是解决前摄干扰问题的关键。

### 为什么必须有梦境

来自 SleepGate 论文（arXiv:2603.14517）的核心发现：

> 旧的过时信息留在 context 里，会**主动干扰**对新信息的正确检索。
> 检索准确率随过时信息积累对数线性下降——无论 context window 多大都无法回避。

典型场景：用户换了工作、改了饮食习惯，但旧信息还在语义记忆里——新信息被错误抑制。

**忘掉旧的矛盾信息，比记住新信息更难，也更重要。**

### 梦境模块的职责

梦境做的是**重写（Rewrite）**，而非追加（Append）：

```
输入：被压缩的对话片段 + 当前 user_profile + 当前长期记忆索引

执行：
  1. 矛盾检测：扫描新内容与已有记忆的冲突（如"用户换工作了"）
  2. 主动覆盖：用新信息覆盖旧的矛盾条目（不是追加，是替换）
  3. 去重：合并重复记忆条目
  4. 去时效：删除"昨天""上周"类相对时间表述，换成绝对时间
  5. 提炼精华：更新 user_profile.json（语义记忆）
  6. 维护精简：user_profile 控制在 < 500 tokens
```

### 触发时机

- **轻量触发**：每次短期记忆压缩后，对被压缩的片段异步执行
- **深度触发**：用户不活跃超过一定时间后，对全部近期记忆执行深度整理
- **定时触发**：每天固定时间（如凌晨）做一次全局整理（二期实现）

### 一期实现范围

一期实现**轻量触发**：压缩时提取 → 矛盾检测 → 覆盖更新 → user_profile 精炼。
深度触发和定时整理推迟到二期。

---

## 六、长期记忆（情景记忆）设计

### 两层存储

```
Qdrant（索引层）                    文件系统（数据层）
  向量 + metadata                     data/memory_store/YYYY-MM.jsonl
  支持语义检索                        append-only，永不修改
  支持时间/类型 filter                Qdrant 挂了可从文件重建索引
  ← 是索引                           ← 是 source of truth
```

### Qdrant 数据结构

```json
{
  "id": "uuid",
  "vector": [/* 2048维 */],
  "payload": {
    "content": "用户提到他不喜欢吃香菜",
    "context_desc": "2026年4月讨论午餐时，用户提到的饮食偏好",
    "memory_type": "episodic",
    "importance": 0.7,
    "created_at": 1712345678,
    "last_accessed_at": 1712345678,
    "access_count": 1,
    "source_file": "data/memory_store/2026-04.jsonl",
    "source_line": 42
  }
}
```

### 写入流程

```
1. LLM 评估重要性（0~1）和类别（事件 / 偏好 / 习惯 / 关系）
2. 重要性 > 0.5 → 进入写入流程：
   a. LLM 生成 context_desc（Contextual Retrieval，提升召回率 67%）
   b. Embed("[context_desc]\n[content]")
   c. 写入 Qdrant
   d. Append 原文到 data/memory_store/YYYY-MM.jsonl
3. 类别是偏好/习惯/关系 → 额外：提交梦境模块更新 user_profile
```

### BM25 索引生命周期

```
启动时：遍历 data/memory_store/*.jsonl，加载所有 content，内存中构建索引
新写入：增量 append 到内存索引，不重建
重启后：从文件全量重建（< 1s，记忆量 < 10万条时）
```

---

## 七、检索模块（三路并行 + 融合重排）

### 检索流程

```
Query
  ├── Embedding → Qdrant 向量检索 → top-20 候选（语义相似）
  ├── BM25 内存索引 → top-20 候选（精确匹配人名/地名/数字）
  └── 二期加入：Graphiti 图谱检索（关系推理）
          ↓
       合并去重
          ↓
       融合打分（Stanford Generative Agents 方案）
          ↓
       top-K 结果
```

### 融合打分

```
score = α × semantic_score
      + β × bm25_score
      + γ × recency_score
      + δ × importance_score

recency_score   = exp(-λ × Δt天数)          λ=0.01
importance_score 每次命中后 +0.05（Hebbian 强化，回写 Qdrant payload）

α=0.4, β=0.3, γ=0.2, δ=0.1（一期默认，后续可调）
```

---

## 八、查询路由

三档规则路由，不上分类模型（三期再升级）：

| 档位 | 触发条件 | Context 构成 | 预期延迟 |
|------|---------|------------|---------|
| 🟢 直接反应 | 指令类、计算、简单闲聊 | System Prompt + 最近 3 轮 | < 100ms |
| 🟡 浅层检索 | 默认档位 | + user_profile + 短期记忆全部 | 200~400ms |
| 🔴 深度检索 | 含回忆类关键词、人名、情感事件 | + 向量+BM25 融合检索 top-K | < 2s |

深度触发词：`上次、之前、以前、记得吗、你还记得、我说过、我提过、上回、那时候`

**动态升级**：
- 浅层回复出现"我不太确定你之前..."→ 自动升级深度补充检索
- 用户追问"更早的呢"→ 强制升级
- 用户打字时预判意图，提前启动检索（降低感知延迟）

---

## 九、Context 组装与 Token 预算

```
System Prompt（程序记忆，固定）          ≤ 800  tokens
user_profile 摘要（语义记忆，全量）      ≤ 500  tokens   ← 梦境模块负责维护精简
长期记忆检索结果（情景记忆，深度档位）   ≤ 1500 tokens
短期记忆（工作记忆，动态）               ≤ 剩余预算
当前用户输入                             ≤ 500  tokens
LLM 回复预留                            ≤ 2000 tokens
```

短期记忆超出剩余预算时，从最旧的摘要段开始丢弃（原始信息已在长期记忆中有备份）。

---

## 十、工具输出压缩（必须实现）

家庭管家场景会大量调用工具（查天气、控制设备、查日历），不压缩会快速撑爆 context。

对应 ReMe `compact_tool_result`：
```
工具返回值 > 200 tokens
  → 写入 data/memory_files/<uuid>.txt
  → Context 里替换为：[工具摘要] + 文件引用路径
```

---

## 十一、代码目录结构

```
AI_Butler/
├── docs/
│   └── tech_spec.md
├── src/
│   ├── main.py                      ← CLI 入口，对话主循环（LangGraph）
│   ├── config.py                    ← 环境变量读取
│   ├── prompts/
│   │   └── system.txt               ← 程序记忆（人格定义）
│   ├── hooks/                       ← ReMe pre_reasoning_hook 四步流程
│   │   ├── compact_tool_result.py   ← Step1：工具输出压缩
│   │   ├── check_context.py         ← Step2：Token 计数
│   │   ├── compact_memory.py        ← Step3：短期记忆部分压缩
│   │   └── summary_memory.py        ← Step4：异步沉淀到长期记忆
│   ├── dream/
│   │   └── dream_agent.py           ← 梦境模块：矛盾检测+覆盖+精炼
│   ├── memory/
│   │   ├── working.py               ← 工作记忆：结构管理 + 落盘
│   │   ├── episodic.py              ← 情景记忆：Qdrant + 文件读写
│   │   ├── semantic.py              ← 语义记忆：user_profile.json
│   │   ├── retriever.py             ← 混合检索 + 融合打分
│   │   └── writer.py                ← 记忆写入决策（重要性判断+分流）
│   ├── router/
│   │   └── query_router.py          ← 三档路由规则
│   ├── context/
│   │   └── assembler.py             ← Context 组装 + token 预算管理
│   └── llm/
│       ├── client.py                ← LLM 调用封装（流式+非流式）
│       └── embedding.py             ← Embedding 调用封装
├── data/                            ← 运行时数据（不进 git）
│   ├── user_profile.json
│   ├── sessions/                    ← 短期记忆落盘
│   ├── memory_store/                ← 长期记忆原文（按月 jsonl）
│   ├── memory_files/                ← 工具输出压缩存储
│   └── qdrant_storage/              ← Qdrant 数据目录
├── docker-compose.yml
├── requirements.txt
├── .env
└── .env.example
```

---

## 十二、各阶段边界

### 一期（当前）
- ReMe 四步 hook 跑通
- 短期记忆：混合结构 + 压缩触发 + 全量落盘
- 长期记忆：Qdrant + 文件两层存储 + 写入流程
- 梦境模块：轻量触发版（压缩时异步执行，矛盾检测+覆盖+精炼）
- 混合检索：向量 + BM25 + 融合打分
- 查询路由：三档规则
- 工具输出压缩：必须实现

### 二期
- 梦境模块：深度触发 + 定时整理（全局重写）
- Graphiti 时序图谱（关系推理能力）
- 数学衰减完整实现（Shodh-Memory 思路）
- 梦境定时任务（每天凌晨）

### 三期
- 查询路由升级为轻量分类模型
- 预判预取（打字时提前检索）
- Prompt 自优化（LangMem 方案）

---

## 十三、一期里程碑

- [ ] **M1**：ReMe 底座集成，四步 hook 结构跑通
- [ ] **M2**：LLM + Embedding 客户端封装，接口联调
- [ ] **M3**：Qdrant 初始化 + 长期记忆两层存储跑通
- [ ] **M4**：BM25 内存索引 + 混合检索 + 融合打分跑通
- [ ] **M5**：短期记忆混合结构 + 压缩触发 + 全量落盘
- [ ] **M6**：梦境模块轻量版（矛盾检测 + 覆盖 + user_profile 精炼）
- [ ] **M7**：查询路由 + Context 组装 + Token 预算管理
- [ ] **M8**：端到端 CLI 对话跑通

---

## 十四、关键参考

| 资料 | 用途 |
|------|------|
| Stanford Generative Agents | 记忆流 + recency×importance×relevance 检索打分理论基础 |
| ReMe（agentscope-ai/ReMe） | 工程底座，四步 hook 直接使用 |
| SleepGate 论文（arXiv:2603.14517） | 前摄干扰理论，梦境模块设计依据 |
| Claude Code AutoDream | 梦境 Agent 工程参考（重写而非追加） |
| Anthropic Contextual Retrieval | 存储时加上下文描述，召回率 +67% |
| LangMem | Hot/Background 分离，Prompt 自优化（三期参考） |
| Shodh-Memory | 算法衰减思路（数学公式，不用 LLM） |
| Graphiti（getzep） | 二期时序图谱 |

---

*v0.3 — 2026-04-06，基于 4月3号完整设计讨论重写，纳入 ReMe 底座、梦境模块核心地位、前摄干扰处理、两层存储、BM25生命周期*
