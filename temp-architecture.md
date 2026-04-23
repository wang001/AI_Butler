# OpenClaw vs Nanobot 架构对比

> 通过 DeepWiki 生成 | 2026-04-23

---

## 一、OpenClaw 架构

**定位：** 多平台 AI 助手系统 | **语言：** TypeScript / Node.js

### 核心组件

```
┌─────────────────────────────────────────────────────┐
│                    Gateway（中心）                     │
│          Node.js 长驻进程 · 协调所有子系统               │
├──────────┬──────────┬──────────┬─────────────────────┤
│ Plugins  │ Channels │ Agents   │ Clients             │
│ 插件系统  │ 消息通道  │ 代理运行时│ 客户端               │
└──────────┴──────────┴──────────┴─────────────────────┘
```

### 系统分层

| 层级 | 技术 | 职责 |
|------|------|------|
| **Gateway** | Node.js / TS | 编排中心、协议、注册表 |
| **Plugins** | 能力模型 | 扩展：通道、模型提供方、工具 |
| **Channels** | 插件 SDK | 消息集成（Telegram / WhatsApp / Discord / Signal 等） |
| **Agents** | 隔离运行时 | 执行助手逻辑（Pi Agent Core） |
| **Tools / Skills** | SDK + Markdown | 系统 / API 操作 + 模块化指导 |
| **Clients** | WebSocket | 用户 / 设备交互面（CLI / Web / 原生 App） |
| **Security** | 信任锚 | 操作者为中心、沙盒、配对机制 |

### 消息生命周期

```
Channel Plugin → Gateway → Agent Runtime → Tool/Provider → Agent → Gateway → Channel
```

### 关键设计

- **插件能力模型**：`registerProvider` / `registerChannel` 等契约，核心与插件边界清晰
- **会话隔离**：每个 Agent 有独立的工作区、模型配置、工具访问
- **四层信任边界**：外部通道 → Gateway 验证 → Agent 隔离 → 工具沙盒
- **Skills 系统**：Markdown 指令文件，教会 Agent 何时 / 如何使用工具链
- **Multi-Agent Routing**：按用户 / 线程 / 会话路由到不同 Agent

---

## 二、Nanobot（HKUDS/nanobot）架构

**定位：** 超轻量 AI Agent 框架 | **语言：** Python | **代码量：** ~4000 行

### 六层架构

```
┌──────────────────────────────────────────────────┐
│  User Interaction    │ CLI 入口 · 命令路由           │
├──────────────────────────────────────────────────┤
│  Communication       │ MessageBus · ChannelManager  │
├──────────────────────────────────────────────────┤
│  Core Intelligence   │ AgentLoop · ContextBuilder   │
│                      │ MemoryStore · SkillsLoader   │
├──────────────────────────────────────────────────┤
│  LLM Integration     │ ProviderRegistry · 多模型     │
├──────────────────────────────────────────────────┤
│  Data & State        │ SessionManager · Cron        │
│                      │ HeartbeatService             │
├──────────────────────────────────────────────────┤
│  Infrastructure      │ Config · ToolRegistry · Bus   │
└──────────────────────────────────────────────────┘
```

### MessageBus（消息总线）

异步 pub/sub 模式，两条独立 `asyncio.Queue`：

- **Inbound Queue**：Channel / Cron / Heartbeat 发布 → AgentLoop 消费
- **Outbound Queue**：AgentLoop / Subagent 发布 → ChannelManager 消费并投递

### AgentLoop（代理循环）

核心处理引擎，执行循环直到生成最终回复或达到 `max_tool_iterations`（默认 200）：

1. 从 MessageBus 接收消息
2. ContextBuilder 组装上下文（历史 + 记忆 + Skills + 运行时信息）
3. 调用 LLM
4. 执行 Tool Calls
5. 结果写回 MessageBus

### 两阶段记忆系统

| 阶段 | 组件 | 功能 |
|------|------|------|
| **Stage 1** | Consolidator | 上下文过大时摘要旧对话，写入 `history.jsonl` |
| **Stage 2** | Dream | Cron 定时触发，深度整理 `MEMORY.md` / `SOUL.md` / `USER.md` |

### MCP 集成

- 支持 Stdio / SSE / Streamable HTTP 三种传输模式
- 连接后自动列出 MCP 工具并注册到 ToolRegistry
- 可按需启用 / 禁用特定工具

### 多实例隔离

- 每个实例有独立的 `workspace`、`config`、`channels`
- Gateway 端口可配置，互不干扰

---

## 三、对比总结

| 维度 | OpenClaw | Nanobot |
|------|----------|---------|
| **语言** | TypeScript（Node.js） | Python |
| **架构风格** | Gateway 中心 + 插件系统 | MessageBus 异步 pub/sub |
| **扩展机制** | 插件能力模型（注册契约） | Channel + Tool 注册表 |
| **记忆系统** | 文件 + 语义搜索 | 两阶段（Consolidator + Dream） |
| **多实例** | 单 Gateway 多 Agent 路由 | 多实例隔离（独立 workspace / port） |
| **客户端** | WebSocket + 原生 App + CLI | CLI 为主，可嵌入 Python 应用 |
| **消息通道** | 插件化，丰富的平台支持 | BaseChannel 抽象（Telegram / Feishu / Matrix） |
| **定位** | 生产级多平台助手系统 | 轻量级可嵌入 Agent 框架 |

### 共同设计理念

- 都采用了 **AgentLoop** 核心循环
- 都有 **Skills** 系统指导 Agent 行为
- 都使用 **文件化记忆**（MEMORY.md / SOUL.md 等）
- 都有 **Cron / Heartbeat** 自主任务机制

### 差异总结

- **OpenClaw** 更偏向 **生产级平台**：丰富的通道集成、多客户端、安全沙盒、完整的插件生态
- **Nanobot** 更偏向 **轻量可嵌入框架**：4000 行代码、Python 生态、MCP 优先、适合二次开发
