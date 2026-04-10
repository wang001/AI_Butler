"""
gateway/web.py — 通用 HTTP / WebSocket 接口

提供面向 Web 前端或第三方调用的 REST + WebSocket API。
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from gateway.server import get_butler

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_key: str = "web:default"


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """单轮问答接口（REST）。"""
    reply = await get_butler().chat(req.message)
    return ChatResponse(reply=reply)


@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    """流式对话接口（WebSocket），每次收到消息返回一条回复。"""
    await websocket.accept()
    butler = get_butler()
    try:
        while True:
            text = await websocket.receive_text()
            reply = await butler.chat(text)
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        pass
