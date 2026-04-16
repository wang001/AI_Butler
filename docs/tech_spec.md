# AI Butler 技术方案（现状版）

> 基于 2026-04-15 代码审查

---

## 一、工程架构

### 1.1 目录结构

```
src/
├── ai_butler.py              CLI/Gateway 统一入口，AIButlerApp 类
├── config.py                Config dataclass，env 加载
├── history.py              ChatHistory，SQLite FTS5 对话日志
├── agent/                  核心 Agent 包（已拆分）
│   ├── __init__.py         Butler 类，薄协调层
│   ├── runner.py           AgentRunner，LLM ↔ Tool 执行循环
│   ├── context.py         ContextBuilder，上下文组装
│   ├── memory.py         MemoryManager，ReMe 封装
│   └── hooks.py          AgentHook / CompositeHook
├── tools/                  工具层
│   ├── __init__.py        ToolDispatcher 导出
│   ├── dispatcher.py     工具分发（if-elif）
│   ├── memory.py       记忆工具
│   ├── search.py      web_search
│   ├── file_reader.py read_file
│   ├── command.py    run_command
│   └── browser.py    browser_use
├── channels/               渠道层
│   ├── cli.py         CLI Channel
│   ├── feishu.py     飞书 Channel
│   └── web.py         Web Channel
├── gateway/               FastAPI 服务
├── cli/                   CLI 辅助（stream 渲染）
├── cron/                  定时任务
│   └── __init__.py    MemoryUpdateService
├── skills/                技能系统（空）
└── prompts/
    └── system.txt     程序记忆
```

### 1.2 核心数据流

```
Channel (CLI/Web/飞书)
    ↓ app.send() / app.send_stream()
AIButlerApp.inbox (asyncio.Queue)
    ↓ _agent_loop 串行消费
Butler.chat() / chat_stream()
    ↓ ContextBuilder.build_context()
    ↓ AgentRunner.run() / run_stream()
LLM + Tools
    ↓ 回复
Channel 输出
```

---

## 二、外部依赖

| 组件 | 选型 | 说明 |
|------|------|------|
| **记忆框架** | ReMe（reme.reme_light.ReMeLight） | 短期记忆压缩 + 长期记忆沉淀 + ChromaDB 向量检索 |
| **对话** | AgentRunner（自研） | 直接调用 LLM，无 LangGraph |
| **LLM** | OpenAI 兼容 API | 百度千帆 / OpenRouter 等 |
| **Embedding** | OpenAI 兼容 API | nvidia/llama-nemotron 等 |
| **历史存储** | SQLite FTS5 | ChatHistory 类，全文检索 |
| **定时任务** | MemoryUpdateService | 后台 6h 定时更新 MEMORY.md |

### 环境变量

```env
LLM_BASE_URL=https://qianfan.baidubce.com/v2/...
LLM_API_KEY=...
LLM_MODEL=glm-5

EMB_BASE_URL=https://openrouter.ai/api/v1
EMB_API_KEY=...
EMB_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free

DATA_DIR=/data
MEMORY_DIR=/data/memory
TOOL_CALL_DIR=/data/tool_call
WORKSPACE_DIR=/workspace

MEMORY_SIMILARITY_THRESHOLD=0.5
MAX_INPUT_LENGTH=128000
COMPACT_RATIO=0.7

COMMAND_ENABLED=true
BROWSER_ENABLED=true
```

---

## 三、模块详解

### 3.1 AIButlerApp（ai_butler.py）

- `inbox: asyncio.Queue` — 所有 Channel 消息入口
- `_agent_loop()` — 串行消费，保证 Butler 状态一致
- 支持 CLI / Gateway 两种模式
- 启动时初始化 Butler，处理关闭时的记忆沉淀

### 3.2 Butler（agent/__init__.py）

薄协调层，组合以下模块：

```python
class Butler:
    def __init__(self, cfg, memory, history, dispatcher, runner, context_builder):
        self._cfg = cfg
        self._memory = memory
        self._history = history
        self._dispatcher = dispatcher
        self._runner = runner
        self._context_builder = context_builder
        self._messages = []           # 当前会话对话历史
        self._compressed_summary = "" # 压缩摘要（跨轮次）
```

核心方法：
- `chat(user_input)` — 非流式对话
- `chat_stream(user_input)` — 流式对话，yield token + tool event

### 3.3 AgentRunner（agent/runner.py）

LLM ↔ Tool 执行循环。

职责：
1. LLM 调用（含 429 限流指数退避重试）
2. Tool Call 循环 — 非流式
3. Tool Call 循环 — 流式
4. 工具批处理（按并发安全性分组）
5. 工具标记过滤（`<|tool_calls_section_begin|>` 等）
6. AgentHook 生命周期事件通知

**⚠️ 已知问题**：流式版本先 `stream=False` 探测是否有 tool_calls，再 `stream=True` 重发，导致无工具调用时每轮 2 次 LLM 调用。

### 3.4 ContextBuilder（agent/context.py）

上下文组装：

```
1. 追加用户消息
2. ReMe pre_reasoning_hook（短期记忆压缩 + 异步沉淀）
3. 静默检索长期记忆（passive_recall）
4. 读取 MEMORY.md
5. 拼装 messages
```

**⚠️ 未实现**：token 预算管理（tech_spec.md 原设计）。

### 3.5 MemoryManager（agent/memory.py）

ReMe 封装：

```python
class MemoryManager:
    async def pre_reasoning_hook(messages, ...) # 短期记忆压缩
    async def passive_recall(query) -> list[str] # 向量检索，相似度过滤
    def read_memory_md() -> str # 读取 MEMORY.md
    async def settle(messages) # 退出时沉淀
    async def close()
```

### 3.6 ToolDispatcher（tools/dispatcher.py）

**⚠️ 已知问题**：使用 if-elif 分发，扩展需修改核心代码。

可用工具（根据配置动态）：
- `search_memory` — ReMe 语义检索
- `search_history` — SQLite FTS5 搜索
- `get_current_time`
- `web_search`
- `read_file`
- `run_command`（需 COMMAND_ENABLED）
- `browser_use`（需 BROWSER_ENABLED）

### 3.7 ChatHistory（history.py）

SQLite FTS5 对话日志：
- WAL 模式，读写不阻塞
- 类级别 threading.Lock
- `append(role, content)` / `search(query, limit)`

### 3.8 MemoryUpdateService（cron/__init__.py）

后台定时任务：
- 6h 间隔读取最新 history
- 独立调用 LLM 判断是否更新 MEMORY.md
- 限制 MEMORY.md ≤ 2000 字

---

## 四、与原设计文档的偏差

| 原设计文档 | 现状 |
|-----------|------|
| `hooks/` 4个 hook 文件 | 已拆分为 `agent/` 包，但完全委托 ReMe |
| `dream/dream_agent.py` | **未实现** |
| `memory/` 5个文件 | 委托给 ReMe + 简单 MEMORY.md |
| `router/query_router.py` 三档路由 | **未实现**，所有查询走同一流程 |
| `context/assembler.py` token预算 | 简化为字符串拼接，无预算管理 |
| LangGraph 状态机 | **未使用**，自研 while 循环 |
| Qdrant + BM25 混合检索 | 委托给 ReMe (ChromaDB) |
| 工具注册表模式 | if-elif 分发 |

---

## 五、已识别的架构问题（P0）

1. **Butler 职责过重** — 虽然已拆分，但仍有 `_messages`、`_compressed_summary` 在 Butler 内部
2. **工具 if-elif 分发** — 扩展需修改 dispatcher.py
3. **流式双重 LLM 调用** — 无 tool_calls 时每轮 2 次请求
4. **无会话持久化** — 进程重启丢失所有状态
5. **token 预算缺失** — ContextBuilder 不感知 context window

---

## 六、配置目录（容器内）

```
/data                          # 内部持久化数据
/data/memory                   # MEMORY.md、chat_history.db
/data/tool_call                # 工具返回值溢出文件
/workspace                    # 命令执行工作区
```

---

*基于 2026-04-15 代码审查重写，反映实际架构*