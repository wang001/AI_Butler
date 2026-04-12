"""
gateway/server.py — FastAPI 应用定义

仅负责：
  1. 持有全局 AIButlerApp 实例（set_app / get_app），供各 Channel 路由调用
  2. 创建 FastAPI app，挂载各 Channel 路由

Butler 的生命周期由 AIButlerApp 统一管理，Channel 路由处理器通过 get_app()
获取 AIButlerApp 实例，再调用 app.send() / app.send_stream() 与 Agent 交互。

挂载路由：
  /        → landingpage/         (网页对话界面静态资源)
  /feishu  → channels/feishu.py   (飞书 Webhook Channel)
  /api     → channels/web.py      (Web Channel: REST / SSE / WebSocket)
  /health  → 健康检查
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from ai_butler import AIButlerApp

# ── 全局 AIButlerApp 实例（由 AIButlerApp._start_gateway 在 uvicorn 启动前注入）──

_app_instance: "AIButlerApp | None" = None


def set_app(app: "AIButlerApp") -> None:
    """由 AIButlerApp._start_gateway() 在 uvicorn.Server.serve() 前调用。"""
    global _app_instance
    _app_instance = app


def get_app() -> "AIButlerApp":
    """供各路由处理器获取 AIButlerApp 实例（含 send() 方法）。"""
    if _app_instance is None:
        raise RuntimeError("AIButlerApp 尚未注入，服务可能还在启动中。")
    return _app_instance


# ── FastAPI 应用 ───────────────────────────────────────────────────────────────

app = FastAPI(title="AI Butler Gateway")

# ── 挂载渠道路由 ───────────────────────────────────────────────────────────────

from landingpage import router as landingpage_router
app.include_router(landingpage_router, tags=["网页界面"])

from channels.feishu import router as feishu_router
app.include_router(feishu_router, prefix="/feishu", tags=["飞书"])

from gateway.web import router as web_router
app.include_router(web_router, prefix="/api", tags=["Web API"])

# ── 健康检查 ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"])
async def health():
    return {"status": "ok", "butler_ready": _app_instance is not None}
