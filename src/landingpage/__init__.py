"""
landingpage — Web 落地页

提供 AI Butler 的网页对话界面。
访问根路径 / 返回 index.html，通过 WebSocket /api/ws 与 Butler 实时对话。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

_STATIC_DIR = Path(__file__).parent

router = APIRouter()


@router.get("/", include_in_schema=False)
async def index():
    """返回网页对话界面。"""
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")
