# AI Butler 技术方案（现状版）

> 基于 2026-04-23 当前代码实现重写。本文描述的是“已经落地的系统”，不是早期设想版。

---

## 一、系统目标

AI Butler 是一个多渠道个人 AI 管家系统，核心目标有三类：

1. 支持 CLI 与 Web 两种主要交互方式
2. 支持带工具调用的 Agent 对话
3. 支持“会话记忆 + 长期记忆”两层状态管理

当前系统已经从“单全局 Butler 串行 inbox”演进为“按会话隔离 runtime + app 级单例长期记忆更新器”的结构。

---

## 二、目录结构

```text
src/
├── ai_butler.py              应用入口，AIButlerApp
├── config.py                 环境变量与运行配置
├── history.py                SQLite 历史与 sessions 元数据
├── agent/
│   ├── __init__.py           Butler 薄协调层
│   ├── runner.py             AgentRunner，LLM ↔ Tool 循环
│   ├── context.py            ContextBuilder，上下文组装
│   ├── memory.py             MemoryManager，ReMe 封装
│   ├── hooks.py              AgentHook / CompositeHook
│   └── stream_events.py      Web 流式事件类型定义
├── channels/
│   ├── cli.py                CLI 渠道
│   └── web.py                Web 渠道（REST / SSE / WebSocket）
├── gateway/                  FastAPI 服务挂载
├── cron/
│   └── __init__.py           app 级单例长期记忆更新器
├── tools/
│   ├── dispatcher.py         ToolDispatcher
│   ├── memory.py             search_memory / search_history / update_memory
│   ├── search.py             web_search
│   ├── search_engine.py      搜索引擎实现
│   ├── web_fetcher.py        网页正文抓取
│   ├── file_reader.py        read_file
│   ├── command.py            run_command
│   └── browser.py            browser_use
├── landingpage/
│   └── index.html            Web UI（会话面板、slash commands、reasoning/tool 卡片）
└── prompts/
    └── system.txt            系统提示词
```

---

## 三、核心架构

### 3.1 分层

系统按 5 层组织：

1. Channel 层  
   CLI / Web 负责接入用户输入和渲染输出

2. App 层  
   `AIButlerApp` 负责 runtime 生命周期、会话恢复、单例服务启动

3. Agent 层  
   `Butler + ContextBuilder + AgentRunner + MemoryManager`

4. Tool 层  
   `ToolDispatcher` 负责 schema 暴露与工具分发

5. Storage / Background 层  
   `ChatHistory(SQLite)` + `MemoryUpdateService`

### 3.2 当前真实数据流

```text
Web/CLI
  ↓
AIButlerApp.send / send_stream / send_event_stream
  ↓
按 conversationId / session_id 获取 runtime
  ↓
runtime.lock 串行保护同一会话
  ↓
Butler.chat / chat_stream
  ↓
ContextBuilder.build_context
  ↓
AgentRunner.run / run_stream
  ↓
LLM + ToolDispatcher
  ↓
history/messages 持久化 + sessions 快照更新
  ↓
单例 MemoryUpdateService 接收“有新消息”通知
```

### 3.3 与旧架构的关键差异

旧设计里最核心的问题是：

- 单个全局 Butler
- Web 不区分独立会话
- 长期记忆更新依赖每个 session 自己的 cron

当前已经改为：

- `session_id -> runtime` 多实例
- 同 session 串行，不同 session 可并发
- `MEMORY.md` 更新由 app 级单例服务统一负责

---

## 四、会话模型

### 4.1 Runtime 结构

`AIButlerApp` 内部维护：

```python
_runtimes: dict[str, _Runtime]
```

其中 `_Runtime` 包含：

- `session_id`
- `channel`
- `butler`
- `lock`
- `last_active_at`

设计含义：

- 同一会话内部严格串行，避免 `_messages` / `_compressed_summary` 被并发覆盖
- 不同会话互不共享 Butler 运行态

### 4.2 会话恢复

Web 会话恢复依赖 `sessions` 表里的快照字段：

- `compressed_summary`
- `summary_history_id`
- `tail_messages_json`

恢复流程：

1. Web 请求带 `conversationId`
2. `AIButlerApp` 查 `sessions`
3. 读取 `compressed_summary + tail_messages_json`
4. 用这些值创建新的 Butler runtime

### 4.3 空闲回收

Web runtime 不是永久常驻的。

当前策略：

- Web 会话空闲超过 `3600s`
- 且当前没有请求正在执行
- 就会触发 runtime eviction

回收时：

- 关闭该 Butler
- 运行态内存释放
- 依靠 `sessions` 表快照支持后续冷恢复

---

## 五、Agent 层设计

### 5.1 Butler

`Butler` 是薄协调层，不直接承担所有细节逻辑。

它主要负责：

- 持有当前会话运行态：
  - `_messages`
  - `_compressed_summary`
- 组织上下文构建
- 调用 AgentRunner
- 持久化本轮 user / assistant / tool_call
- 更新会话快照

### 5.2 ContextBuilder

当前上下文构建顺序是：

1. 把当前用户输入追加进 `_messages`
2. 调用 `MemoryManager.pre_reasoning_hook`
3. 执行 `passive_recall`
4. 读取 `MEMORY.md`
5. 组装最终 messages

当前 system prompt 结构为：

- `system.txt`
- `MEMORY.md` 内容（若存在）
- `history` 里的 user / assistant 文本
- 检索到的记忆片段

### 5.3 AgentRunner

AgentRunner 负责：

- OpenAI 兼容 API 调用
- 429 限流指数退避
- tool call 循环
- 流式事件输出
- provider 兼容处理

当前有两个重要实现特征：

1. provider 兼容  
   对不支持 `system` role 的兼容层，会把 system 内容折叠进第一条 user 消息

2. streaming 仍是“两阶段”  
   流式模式下仍然会先非流式探测 `tool_calls`，无工具时再发起真正的流式回复  
   这意味着“无工具调用的一轮回复”仍可能发生两次 LLM 请求

---

## 六、Web 协议与前端

### 6.1 Web API

当前 Web 层由 [web.py](/Users/wangxianci/opensource/AI_Butler/src/channels/web.py) 提供：

- `POST /api/chat`
- `POST /api/chat/stream`
- `POST /api/sessions`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/messages`
- `WS /api/ws`

### 6.2 流式协议

系统内部与 Web 输出统一采用结构化事件流，事件定义在 [stream_events.py](/Users/wangxianci/opensource/AI_Butler/src/agent/stream_events.py)。

当前协议尽量贴近 AI SDK UI message stream，核心事件包括：

- `start`
- `start-step`
- `reasoning-start`
- `reasoning-delta`
- `reasoning-end`
- `tool-input-start`
- `tool-input-delta`
- `tool-input-available`
- `tool-output-available`
- `text-start`
- `text-delta`
- `text-end`
- `finish-step`
- `finish`
- `error`

SSE 额外带：

```text
x-vercel-ai-ui-message-stream: v1
```

WebSocket 则复用同一套 JSON 帧结构。

### 6.3 前端行为

[landingpage/index.html](/Users/wangxianci/opensource/AI_Butler/src/landingpage/index.html) 当前已经支持：

- 会话面板
- 新建会话
- 切换历史会话
- 恢复当前会话历史
- reasoning 折叠卡片
- tool call / tool result 卡片
- slash commands

当前 slash commands 包括：

- `/new`
- `/clear`
- `/sessions`
- `/session <keyword>`
- `/thinking`
- `/details`
- `/help`

输入 `/` 时会弹出命令提示面板，支持：

- 上下键选择
- `Tab` 补全
- `Enter` 执行

### 6.4 CLI 行为

CLI 现在刻意简化：

- 不展示工具时间线
- 不展示 reasoning
- 只展示最终回复文本

这让 CLI 更像一个轻量 terminal 聊天入口，而不是调试面板。

---

## 七、工具层

### 7.1 ToolDispatcher

`ToolDispatcher` 的职责：

1. 动态构建当前可用工具 schema
2. 按名称分发工具调用
3. 统一参数校验
4. 结果过长时溢出到文件

### 7.2 当前工具集合

固定工具：

- `search_memory`
- `search_history`
- `update_memory`
- `get_current_time`
- `web_search`
- `web_fetcher`
- `read_file`

按配置启用：

- `run_command`
- `browser_use`

### 7.3 参数错误处理

当前工具层已经支持：

- 参数名校验
- 缺少必填参数校验
- `TypeError` 转可恢复文本错误

这样模型在第一次工具调用出错时，不会直接把整轮对话打断。

---

## 八、存储模型

### 8.1 messages 表

`messages` 是事实流水表，字段核心包括：

- `id`
- `ts`
- `role`
- `content`
- `session_id`
- `channel`

用途：

- 完整对话日志
- FTS 搜索源
- 长期记忆更新的增量事实源

当前 `tool_call` 也会写入 `messages`，但不进入 FTS。

### 8.2 sessions 表

`sessions` 是会话目录与快照表，核心字段包括：

- `id`
- `title`
- `channel`
- `status`
- `preview`
- `compressed_summary`
- `summary_history_id`
- `tail_messages_json`
- `created_at`
- `updated_at`
- `last_active_at`
- `last_message_at`

用途：

- 会话列表展示
- 当前会话 subtitle / preview
- 冷恢复运行态

### 8.3 FTS 与并发

当前 SQLite 方案：

- 主库启用 WAL
- 同一 `data_dir` 使用类级别 `threading.Lock`

含义是：

- 读写基本不会互相阻塞
- 同进程内写提交仍采取保守串行策略

这对当前低 QPS 的个人管家场景是可以接受的。

---

## 九、长期记忆设计

### 9.1 两类记忆

系统当前有两套不同用途的记忆：

1. 会话记忆  
   `_messages + _compressed_summary + sessions 快照`

2. 长期记忆  
   `MEMORY.md`

### 9.2 当前长期记忆更新器

当前长期记忆不再由每个 session 各自维护，而是由 `app` 级单例 `MemoryUpdateService` 统一负责。

它的输入不是某个 session 的 `_messages`，而是全局 `messages` 表。

当前策略：

- 只看 `role in ('user', 'assistant')`
- 用 `last_processed_message_id` 作为增量游标
- 用 `last_updated_at` + `1h` 做节流
- 只在“上次更新后有新消息且已到期”时调模型

### 9.3 增量与分批

单次更新默认最多处理 `500` 条消息。

但不是简单“截断 500 条就结束”，而是：

- 固定本轮起跑时的 `target_message_id`
- 按 `500` 条分批推进
- 如果积压超过 `500`，会继续批处理直到追上当前目标 id

这保证：

- 不会因为批次限制丢消息
- 也不会无限追逐更新过程中新增的更晚消息

### 9.4 meta 文件

长期记忆更新器当前使用：

```text
<memory_dir>/.memory_update_meta.json
```

字段包括：

- `version`
- `last_processed_message_id`
- `last_checked_at`
- `last_updated_at`
- `next_due_at`

### 9.5 MEMORY.md 的边界

当前 `MEMORY.md` 仍然是全局共享的，不是 per-session。

这意味着它适合存：

- 用户稳定偏好
- 长期项目
- 关键约定
- 角色身份

而不适合存：

- 某个会话的一次性上下文
- 短期任务中间态

### 9.6 ReMe 的当前角色

本次重构没有移除 ReMe。

当前 ReMe 仍然承担：

- `pre_reasoning_hook`
- `passive_recall`
- `settle`
- `search_memory`

也就是说：

- 长期记忆更新器已经和 per-session cron 解耦
- 但主对话热路径仍然保留 ReMe

---

## 十、运行目录

当前目录约定：

```text
DATA_DIR         内部持久化数据根目录
MEMORY_DIR       记忆与历史目录
TOOL_CALL_DIR    工具结果溢出目录
WORKSPACE_DIR    用户工作区 / 命令执行目录
```

本地开发现在更推荐把这些目录都指向 `workspace/.ai_butler/...`，避免误写仓库根目录或宿主环境其他路径。

---

## 十一、已知问题与后续方向

### 11.1 已知问题

1. AgentRunner 流式路径仍可能发生双请求  
   没有 tool call 的一轮回复，当前仍是“先探测，再真实流式输出”

2. ReMe 仍在主链路热路径  
   系统尚未完成“只靠 history + MEMORY.md”那套更可控的主路径重构

3. `MEMORY.md` 仍是全局文件  
   这符合“个人管家”设定，但不适合未来多用户共享部署

4. ToolDispatcher 仍是名称分发  
   当前可维护，但还不是 registry/plugin 化模型

### 11.2 当前更适合做的后续优化

1. 把长期记忆更新状态可视化  
   Web UI 增加“上次更新时间 / 下次到期时间”

2. 继续收紧长期记忆生成提示  
   减少 `MEMORY.md` 噪音，提升长期画像稳定性

3. 评估是否把 ReMe 从热路径降级  
   但这属于后续架构调整，不在本文当前范围

---

## 十二、总结

到 2026-04-23 为止，AI Butler 的核心架构已经具备以下稳定能力：

- Web 会话隔离与恢复
- AI SDK 风格结构化流式协议
- reasoning / tool call / final text 分轨渲染
- `sessions + messages` 双表持久化
- app 级单例长期记忆更新器
- slash command 驱动的会话操作

当前系统已经不再是“单全局 Agent + 简单网页壳”，而是一个具备多会话、结构化流、长期记忆更新和基础前端状态管理的个人 AI 管家平台。
