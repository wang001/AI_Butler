"""
gateway — 常驻 Web 服务层

启动 FastAPI 服务，将各渠道（channels/）的请求统一路由给 Butler 处理。
与具体渠道实现解耦：渠道模块放在 channels/ 包中，gateway 只负责注册和生命周期。

启动：
  uvicorn gateway.server:app --host 0.0.0.0 --port 8080

模块：
  - server.py : FastAPI app、lifespan（Butler 初始化/关闭）、路由注册、/health
  - web.py    : 通用 REST + WebSocket 接口（供 landingpage 和第三方调用）
"""
