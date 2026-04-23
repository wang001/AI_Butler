"""
channels/web.py — Web 消息渠道（HTTP REST + SSE + WebSocket）

流式协议尽量贴近 AI SDK UI message stream：
  - start / finish
  - start-step / finish-step
  - text-start / text-delta / text-end
  - reasoning-start / reasoning-delta / reasoning-end
  - tool-input-start / tool-input-delta / tool-input-available
  - tool-output-available
  - error

SSE 是对外的 canonical API，额外携带
`x-vercel-ai-ui-message-stream: v1` 以便兼容现有开源生态。
WebSocket 则复用相同 JSON chunk，作为浏览器本地页面的便捷传输层。
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# WebSocket ping 间隔（秒）：防止工具执行期间（可能几十秒）连接被代理/浏览器超时关闭
_WS_PING_INTERVAL = 20

router = APIRouter()


def _frame(data: dict) -> str:
    """序列化 JSON 帧为字符串（WebSocket send_text / SSE data 共用）。"""
    return json.dumps(data, ensure_ascii=False)


def _public_session(session: dict | None) -> dict | None:
    """向前端返回轻量会话摘要，避免把 runtime 快照直接暴露给 UI。"""
    if not session:
        return None
    return {
        "id": session.get("id", ""),
        "title": session.get("title", ""),
        "channel": session.get("channel", "unknown"),
        "status": session.get("status", "active"),
        "preview": session.get("preview", ""),
        "created_at": session.get("created_at", 0),
        "updated_at": session.get("updated_at", 0),
        "last_active_at": session.get("last_active_at", 0),
        "last_message_at": session.get("last_message_at", 0),
    }


# ── 非流式 REST ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversationId: str | None = None


class ChatResponse(BaseModel):
    reply: str


class SessionCreateRequest(BaseModel):
    title: str = "新会话"


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """单轮问答接口（REST，非流式，返回完整回复）。"""
    from gateway.server import get_app
    reply = await get_app().send(
        req.message,
        session_id=req.conversationId,
        channel="web",
    )
    return ChatResponse(reply=reply)


# ── SSE 流式 REST ──────────────────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    SSE 流式问答接口（AI SDK 风格 UI message stream）。

    每条事件：data: <JSON帧>\\n\\n
    流结束：data: [DONE]\\n\\n
    """
    from gateway.server import get_app
    app = get_app()

    async def event_generator():
        try:
            async for event in app.send_event_stream(
                req.message,
                session_id=req.conversationId,
                channel="web",
            ):
                frame = dict(event)
                if req.conversationId:
                    frame["conversationId"] = req.conversationId
                yield f"data: {_frame(frame)}\n\n"
        except Exception as exc:
            yield f"data: {_frame({'type': 'error', 'errorText': str(exc)})}\n\n"
            yield f"data: {_frame({'type': 'finish', 'finishReason': 'error'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "x-vercel-ai-ui-message-stream": "v1",
            "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲，确保实时推送
        },
    )


# ── 会话元接口 ──────────────────────────────────────────────────────────────────

@router.post("/sessions")
async def create_session(req: SessionCreateRequest | None = None) -> dict:
    """创建新的 Web 会话。"""
    from gateway.server import get_app
    app = get_app()
    session = await app.create_session(
        channel="web",
        title=(req.title if req else "新会话"),
    )
    return _public_session(session) or {}


@router.get("/sessions")
async def list_sessions(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """列出最近活跃的 Web 会话。"""
    from gateway.server import get_app
    app = get_app()
    sessions = app.list_sessions(channel="web", limit=limit)
    return {"sessions": [_public_session(session) for session in sessions]}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """读取单个会话元信息。"""
    from gateway.server import get_app
    app = get_app()
    session = app.get_session(session_id)
    return {"session": _public_session(session)}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=500),
) -> dict:
    """读取某个会话的历史消息。"""
    from gateway.server import get_app
    app = get_app()
    return {"messages": app.get_session_messages(session_id, limit=limit)}


# ── WebSocket 流式对话 ─────────────────────────────────────────────────────────

@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    """
    WebSocket 流式对话接口（JSON 帧协议）。

    客户端发送：纯文本（用户输入）
    服务端推送：JSON 帧，见模块文档中的协议说明。

    心跳机制：
      工具调用可能耗时数十秒，期间主动发送 {"type":"ping"} 保活帧，
      防止浏览器 / Nginx / 代理因超时关闭连接。
    """
    await websocket.accept()

    from gateway.server import get_app
    app = get_app()

    async def _stream_with_ping(text: str, conversation_id: str | None) -> None:
        """
        并发执行两个协程：
        - 主流：消费 app.send_event_stream()，将每个事件转为 JSON 帧发送
        - 副流：每隔 _WS_PING_INTERVAL 秒发送一次 {"type":"ping"} 保活帧
        两者通过 done_event 协调退出。
        """
        done_event = asyncio.Event()

        async def _pinger() -> None:
            while not done_event.is_set():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.sleep(_WS_PING_INTERVAL)),
                        timeout=_WS_PING_INTERVAL + 1,
                    )
                    if not done_event.is_set():
                        await websocket.send_text(_frame({"type": "ping"}))
                except Exception:
                    break

        ping_task = asyncio.create_task(_pinger())
        try:
            async for event in app.send_event_stream(
                text,
                session_id=conversation_id,
                channel="web",
            ):
                frame = dict(event)
                if conversation_id:
                    frame["conversationId"] = conversation_id
                await websocket.send_text(_frame(frame))
        except Exception as exc:
            await websocket.send_text(_frame({"type": "error", "errorText": str(exc)}))
            await websocket.send_text(_frame({"type": "finish", "finishReason": "error"}))
        finally:
            done_event.set()
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

    try:
        while True:
            incoming = await websocket.receive_text()
            conversation_id = None
            text = incoming
            try:
                payload = json.loads(incoming)
                if isinstance(payload, dict) and payload.get("type") == "input":
                    text = str(payload.get("text") or "")
                    conversation_id = payload.get("conversationId")
            except Exception:
                pass

            await _stream_with_ping(text, conversation_id)
    except WebSocketDisconnect:
        pass
