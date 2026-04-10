"""
gateway/server.py — 常驻 Web 服务入口

使用 FastAPI 启动 HTTP 服务，在 lifespan 中初始化 Butler，
通过 get_butler() 把实例共享给 channels/ 和 web.py 的路由处理器。

启动方式：
  uvicorn gateway.server:app --host 0.0.0.0 --port 8080

路由挂载：
  - /feishu  → channels/feishu.py  (飞书 webhook)
  - /wecom   → channels/wecom.py   (企微 webhook)
  - /api     → gateway/web.py      (通用 REST / WebSocket)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import Config
from agent import Butler, wait_heavy_loaded

# ── 全局 Butler 实例 ───────────────────────────────────────────────────────────

_butler: Butler | None = None


def get_butler() -> Butler:
    """供各路由处理器获取共享的 Butler 实例。"""
    if _butler is None:
        raise RuntimeError("Butler 尚未初始化，服务可能还在启动中。")
    return _butler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化 Butler，关闭时释放资源。"""
    global _butler
    cfg = Config.from_env()
    wait_heavy_loaded()
    _butler = await Butler.create(cfg, channel="api")
    yield
    if _butler:
        await _butler.close()
    _butler = None


app = FastAPI(title="AI Butler Gateway", lifespan=lifespan)

# ── 挂载渠道路由 ───────────────────────────────────────────────────────────────
# 取消注释以启用对应渠道

from channels.feishu import router as feishu_router
app.include_router(feishu_router, prefix="/feishu", tags=["飞书"])


from gateway.web import router as web_router
app.include_router(web_router, prefix="/api", tags=["Web API"])


# ── 健康检查 ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"])
async def health():
    return {"status": "ok", "butler_ready": _butler is not None}
