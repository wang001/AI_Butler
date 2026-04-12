"""
channels — 消息渠道接入层

每个模块对应一种外部消息渠道，通过 AIButlerApp.send() / send_stream()
与 Agent 通信，各自负责自身的 I/O 和协议差异化。

各渠道共同的调用模式：
    1. channel 初始化自身 I/O 资源
    2. 获取用户输入
    3. 调用 app.send(text) 或 app.send_stream(text) 与 Agent 交互
    4. 通过渠道自身方式将回复输出给用户

渠道列表：
    - cli.py   : 命令行 prompt 读取、ThinkingSpinner、quit 关键字退出 session
    - feishu.py: 飞书 URL 验证握手、签名校验、OpenAPI 回复
    - web.py   : HTTP REST / SSE / WebSocket 接入（网页对话界面及第三方调用）
                 路由由 gateway/server.py 以 prefix="/api" 挂载

HTTP 渠道（feishu / web）由 gateway/server.py 在启动时加载并注册为 FastAPI 路由，
实现 router: APIRouter 即可接入。
"""
