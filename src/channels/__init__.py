"""
channels — 消息渠道接入层

每个模块对应一种外部消息渠道，统一调用 skills.agent.Butler.chat() 处理消息。

各渠道共同的调用模式：
    1. channel 初始化自身 I/O 资源
    2. 获取 Butler 实例（由 gateway 或入口创建）
    3. 调用 await butler.chat(user_input) 获取回复
    4. 通过渠道自身方式输出回复

渠道各自负责的差异化逻辑：
    - cli.py   : 命令行 prompt 读取、ThinkingSpinner、quit 关键字退出 session
    - feishu.py: 飞书 URL 验证握手、签名校验、OpenAPI 回复
    - wecom.py : 企业微信 XML 解析、被动回复加密

HTTP 渠道（feishu / wecom）由 gateway/server.py 在启动时加载并注册为 FastAPI 路由，
实现 router: APIRouter 即可接入。
"""
