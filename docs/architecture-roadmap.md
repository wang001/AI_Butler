# AI Butler 架构演进路线图

> 基于 2026-04-23 当前实现整理。本文不是“现状说明”，而是面向未来 2-3 个阶段重构的目标文档。  
> 参考来源：当前代码实现、[tech_spec.md](/Users/wangxianci/opensource/AI_Butler/docs/tech_spec.md)、[temp-architecture.md](/Users/wangxianci/opensource/AI_Butler/docs/temp-architecture.md) 中对 OpenClaw / Nanobot 的架构对比。

---

## 一、文档目标

这份路线图回答 3 个问题：

1. AI Butler 长期应该演进成什么样的架构
2. 当前实现里哪些结构会阻碍长期演进
3. 后续重构应该按什么顺序推进，避免“只做目录美化，不解决根问题”

本文希望达到的效果不是追求抽象名词，而是给后续代码重构提供明确边界：

- 什么值得现在做
- 什么应该晚一点做
- 什么暂时不要做

---

## 二、参考架构给我们的启发

### 2.1 从 OpenClaw 借什么

OpenClaw 最值得借鉴的不是“大而全”，而是边界清晰：

- Gateway 是明确的编排中心
- Channel / Provider / Tool 是可注册能力，不是核心里硬编码
- Agent Runtime 是隔离的执行单元
- 能力访问有 trust boundary / capability boundary

对 AI Butler 的启发是：

- 我们需要注册表驱动的系统，而不是继续堆 `if/elif`
- 我们需要把“部署方式”“工具能力”“会话状态”解耦
- 我们需要把 App 层做成真正的 composition root

### 2.2 从 Nanobot 借什么

Nanobot 最值得借鉴的是“Python 小核心 + 明确分层”：

- AgentLoop 是唯一核心执行引擎
- Session / Memory / Cron / Registry 都是配套系统，不和 AgentLoop 混写
- 两阶段记忆天然适合 Agent 产品

对 AI Butler 的启发是：

- 我们应该继续收敛到 `Gateway/App + AgentLoop + Stores + Background Services`
- 不需要为了平台化而过早引入复杂插件系统
- 先把 Python 内部结构做好，比盲目追求 feature 数量更重要

### 2.3 适合 AI Butler 的方向

AI Butler 的合理目标不是“OpenClaw 的 Python 复刻”，也不是“Nanobot 的原样照搬”，而是：

- 用 OpenClaw 的注册契约和能力边界
- 用 Nanobot 的轻量分层和 AgentLoop 思路
- 保留我们自己已经成型的 Web 会话、结构化事件流、长期记忆文件方案

一句话说：

> AI Butler 适合演进成一个“轻量但可平台化”的 Python Agent Gateway。

---

## 三、当前架构的主要问题

### 3.1 `AIButlerApp` 正在成为半个上帝类

当前 `AIButlerApp` 既承担：

- runtime 生命周期管理
- session 恢复
- channel 接口
- singleton service 启动
- 部分 composition root 职责

这比旧版已经好很多，但长期还是会继续膨胀。  
后续如果再接更多 channel、provider、heartbeat、automation，App 层会迅速变胖。

### 3.2 `Butler.create()` 仍承担装配职责

当前 `Butler.create()` 自己创建：

- LLM client
- MemoryManager
- ChatHistory
- CommandExecutor
- BrowserAgent
- ToolDispatcher

这会导致：

- `Butler` 仍然不够“纯”
- 单元测试和依赖替换困难
- App 层与 Runtime 层边界不够清晰

### 3.3 `ToolDispatcher` 同时承担太多责任

当前工具层混了几类职责：

- schema 暴露
- 工具启停判定
- 参数校验
- 执行分发
- 并发安全策略
- 结果溢出到文件

这说明问题已经不只是“tools 目录平铺”，而是工具系统本身还没有真正模块化。

### 3.4 `history.py` 既像数据库层，又像业务层

当前 `history.py` 同时负责：

- schema 初始化 / migration
- message append
- session 元数据
- FTS 搜索
- 并发写锁

这会造成 store 层和业务恢复逻辑耦合过深。

### 3.5 事件流已经成型，但内部事件模型还不够统一

当前 Web 协议已经接近 AI SDK 风格，但系统内部还没有一个统一“总线语义”的事件层。

现在的事件流更多是：

- 给 Web UI 用
- 给 AgentRunner 输出用

而不是系统内部统一的状态事件。

后续要做 heartbeat、automation、更多 channel、甚至 subagent 时，这个缺口会放大。

### 3.6 Provider 适配逻辑还不够抽象

当前 provider 兼容处理主要集中在 AgentRunner。

这会造成：

- 模型差异处理越来越多地堆进执行循环
- 流式协议与 provider 行为耦合
- 新接供应商时容易污染 AgentLoop 主链路

### 3.7 测试面仍然偏薄

当前系统已经不再是 demo 规模，但关键 contract 的自动化测试还不成体系。

最关键的风险是：

- 重构工具系统时不敢动
- 重构事件流时靠手点回归
- session 恢复和 memory 更新容易出隐性回归

---

## 四、目标架构

### 4.1 总体目标

目标架构如下：

```text
Clients / Channels
  - Web
  - CLI
  - Future adapters

Gateway / App
  - SessionManager
  - RuntimeFactory
  - ChannelRegistry
  - ProviderRegistry
  - ToolRegistry
  - BackgroundServices

Agent Runtime
  - AgentLoop
  - ContextBuilder
  - MemoryOrchestrator
  - SkillLoader

Stores
  - MessageStore
  - SessionStore
  - MemoryStore
  - ArtifactStore

Capabilities / Policies
  - Tool capability grants
  - Workspace policy
  - Sandbox / trust boundary
```

### 4.2 分层职责

#### Gateway / App 层

负责“编排”，不负责具体推理细节：

- 接收用户输入
- 解析 channel / session / routing
- 装配 runtime
- 启动后台服务
- 暴露 API

#### Agent Runtime 层

负责“执行一轮 Agent 推理循环”：

- 构建上下文
- 调用模型
- 执行工具
- 维护会话运行态

它不应该直接关心：

- FastAPI
- WebSocket
- SQLite schema 初始化
- runtime 装配细节

#### Stores 层

负责“状态持久化”，不负责业务编排：

- 原始消息事实
- 会话快照
- 记忆文件
- artifact / tool spill 文件

#### Capability / Policy 层

负责“是否允许”和“边界在哪”：

- 哪些工具对哪个 runtime 可用
- workspace 范围
- sandbox / browser / shell 风险边界

这层是未来产品安全性的重要基础。

---

## 五、目标设计原则

后续所有重构，建议围绕下面 6 条原则判断是否值得做。

### 5.1 契约优先

能定义成 contract 的，不靠字符串约定。

重点包括：

- ToolSpec
- StreamEvent
- SessionSnapshot
- ProviderAdapter

### 5.2 状态边界清晰

必须明确区分：

- runtime state
- persisted session state
- long-term memory
- history facts
- temporary artifacts

不要让同一份数据既像 cache，又像 source of truth。

### 5.3 装配与业务解耦

谁负责“创建对象”，谁负责“执行业务”，边界要清楚。

目标状态：

- App / RuntimeFactory 负责装配
- AgentLoop / Butler 负责运行

### 5.4 可恢复优先于可变魔法

对于 session / memory / tool 这类状态系统，优先选择：

- 行为可解释
- 状态可落盘
- 重启后能恢复

而不是过度依赖运行时隐式状态。

### 5.5 背景任务必须是 sidecar，而不是主链路污染源

长期记忆更新、heartbeat、automation 都应该是后台服务，不应该把主对话执行链弄脏。

### 5.6 重构必须带 contract tests

每次架构性改动，至少补一类测试：

- tool contract test
- stream event test
- session recovery test
- memory update test

---

## 六、建议的模块演进方向

### 6.1 App 层：引入 `SessionManager` 与 `RuntimeFactory`

建议把当前 `AIButlerApp` 继续拆成：

- `SessionManager`
  - runtime 缓存
  - idle eviction
  - session 元数据读取
- `RuntimeFactory`
  - 装配 Butler runtime
  - 注入 provider / tools / stores / background service references

这样 `AIButlerApp` 本身只保留：

- API-facing 方法
- 服务生命周期
- registry 挂载

### 6.2 Agent 层：把 `Butler` 收敛成真正的 runtime facade

建议目标是：

- `Butler` 只保留跨轮会话状态和入口方法
- `AgentLoop` 负责一次推理循环
- `ContextBuilder` 只管构造 prompt/messages
- `MemoryOrchestrator` 管理 ReMe + MEMORY.md + recall / settle

如果未来 `Butler` 继续膨胀，就说明边界又在回退。

### 6.3 Tool 层：从 `ToolDispatcher` 升级到 `ToolRegistry`

建议目标结构：

```text
src/tools/
  registry.py
  types.py
  web/
  system/
  memory/
```

核心对象建议统一成：

```python
ToolSpec(
    name=...,
    schema=...,
    handler=...,
    concurrent_safe=...,
    enabled_when=...,
    spill_policy=...,
)
```

这样可以把当前 `dispatcher.py` 的多重职责拆开：

- Registry：注册与发现
- Executor：执行与错误包装
- Policy：启停与并发安全
- Serializer：tool result 输出格式

### 6.4 Provider 层：引入 `ProviderRegistry` + `ProviderAdapter`

建议把模型兼容行为抽成 provider adapter：

- system role 不支持时如何降级
- tool call 流式如何解析
- reasoning delta 如何提取
- finish reason 如何归一化

这样 `AgentRunner` 就能只关心“统一后的 provider contract”。

### 6.5 Store 层：拆分 `history.py`

建议拆为：

- `MessageStore`
- `SessionStore`
- `HistorySearchStore`
- `MigrationManager`

这会让“事实流水”“会话恢复”“检索”三种职责更清楚。

### 6.6 Background Services：显式服务目录

建议未来统一放在 `services/` 或 `background/`：

- `MemoryUpdateService`
- `HeartbeatService`
- `AutomationService`
- 未来的 `NotificationService`

这样比继续把所有后台任务塞进 `cron/` 更稳定。

---

## 七、建议的分阶段实施计划

### Phase 1：收敛核心边界

这是最值得优先做的一期。

目标：

- 引入 `ToolSpec` / `ToolRegistry`
- 把工具目录按领域拆包
- 引入 `RuntimeFactory`
- 让 `Butler.create()` 不再负责依赖装配

完成标志：

- 新增工具不需要改 `if/elif`
- 工具启停和 schema 暴露由注册表统一处理
- `Butler` 构造时接受依赖，而不是自己拼依赖

### Phase 2：抽象 provider 与 store

目标：

- 引入 `ProviderAdapter`
- 抽离 `ProviderRegistry`
- 拆分 `history.py`
- 明确 `MessageStore / SessionStore / SearchStore`

完成标志：

- 接新模型供应商时主要改 provider adapter
- session 恢复逻辑不再依赖大而全的 `ChatHistory`

### Phase 3：统一事件与后台系统

目标：

- 定义进程内统一事件语义
- heartbeat / automation / background jobs 统一接入
- stream event 与 internal event 建立映射关系

注意：

这里不是要求立刻上重型 MessageBus，而是先把事件 contract 统一。

完成标志：

- 新增后台服务时不需要侵入 App 主链路
- Web / CLI / future channels 共享更一致的事件模型

### Phase 4：平台化能力建设

这一阶段不急，但值得作为长期方向保留。

目标：

- capability grants
- per-runtime tool visibility
- plugin-like registration SDK
- more channels / more clients / MCP-like adapters

这一步应该建立在前 3 期边界已经稳定的基础上，而不是现在直接冲。

---

## 八、明确的非目标

为了避免路线图失控，下面这些事不建议现在做。

### 8.1 不立刻做完整插件市场

我们需要的是注册契约，不是插件生态运营。

### 8.2 不立刻做分布式消息总线

当前进程内一致事件模型就够了，没必要为了“看起来高级”引入外部队列。

### 8.3 不立刻做多 Agent 编排平台

当前主需求还是单用户个人管家。  
把单 runtime、session、memory、tool 边界打稳，比过早搞 subagent orchestration 更值。

### 8.4 不立刻抛弃 ReMe

是否替换 ReMe 可以继续评估，但这不该阻塞 App / Agent / Tool / Store 的架构重构。

---

## 九、每一期都应该补的测试

### Phase 1 至少要补

- tool schema 与 handler 对齐测试
- tool registry enable/disable 测试
- spill-to-file 行为测试

### Phase 2 至少要补

- provider adapter contract tests
- session recovery tests
- message / session store integration tests

### Phase 3 至少要补

- stream event ordering tests
- memory updater incremental drain tests
- background service wakeup / throttle tests

---

## 十、对当前代码的映射建议

### 当前 → 未来模块映射

- `src/ai_butler.py`
  - 未来拆向 `app.py`、`session_manager.py`、`runtime_factory.py`

- `src/agent/__init__.py`
  - 未来更像 `runtime.py` 或 `butler.py`

- `src/agent/runner.py`
  - 保留为 AgentLoop 核心，但要减少 provider 兼容分支

- `src/history.py`
  - 未来拆成多个 store

- `src/tools/dispatcher.py`
  - 未来拆成 registry / executor / policy

- `src/cron/__init__.py`
  - 未来迁到 `src/services/memory_update.py`

- `src/channels/web.py`
  - 可继续保留，但应更多依赖 app contracts，而不是细节对象

---

## 十一、架构成熟度判断标准

当我们满足下面这些条件时，可以认为 AI Butler 的架构已经从“能跑”进入“长期可演进”：

- 新增一个工具，不需要改 3 个以上核心文件
- 新增一个模型供应商，不需要侵入 AgentLoop 主链路
- 新增一个后台服务，不需要让 App 层继续膨胀
- session 恢复、memory 更新、stream 协议都有自动化回归
- 目录结构和职责分层对新同学来说是可解释的

---

## 十二、建议的下一步

如果按这份路线图推进，我建议下一步直接从 Phase 1 开始，不先做大重写。

最推荐的首个切入点是：

1. 定义 `ToolSpec`
2. 引入 `ToolRegistry`
3. 把 `tools` 目录按 `web/system/memory` 拆包
4. 让 `dispatcher.py` 退化成执行器，而不是总调度中心

这是当前投资回报最高、对后续 provider/store/session 重构帮助最大的第一枪。
