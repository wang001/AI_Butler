"""
gateway/server.py — FastAPI 应用定义

不再负责 Butler 的生命周期管理（由 AIButlerApp 统一持有）。
路由处理器通过 get_app() 获取 AIButlerApp 实例，
再调用 app.send(text) 向 Agent 主循环提交消息。

挂载路由：
  /feishu  → channels/feishu.py  (飞书 webhook)
  /api     → gateway/web.py      (通用 REST / WebSocket)
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

from channels.feishu import router as feishu_router
app.include_router(feishu_router, prefix="/feishu", tags=["飞书"])

from gateway.web import router as web_router
app.include_router(web_router, prefix="/api", tags=["Web API"])

# ── 健康检查 ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"])
async def health():
    return {"status": "ok", "butler_ready": _app_instance is not None}
