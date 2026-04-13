"""
channels/web.py — Web 消息渠道（HTTP REST + SSE + WebSocket）

统一的 Web Channel，遵循与 cli.py / feishu.py 相同的 Channel 约定：
  - 接收来自浏览器的用户输入
  - 通过 app.send() / app.send_stream() 与 Agent 通信
  - 将 Agent 的流式 token 翻译为 JSON 帧，通过 WebSocket / SSE 推送给前端

路由说明（在 gateway/server.py 以 prefix="/api" 挂载）：
  POST /api/chat           非流式问答（返回完整 JSON）
  POST /api/chat/stream    SSE 流式问答（text/event-stream）
  WS   /api/ws             WebSocket 流式对话（JSON 消息帧）

─── WebSocket 消息协议 ────────────────────────────────────────────────────
  客户端 → 服务端：纯文本（用户输入）

  服务端 → 客户端：JSON 消息帧，type 字段区分：
    {"type": "token",       "text": "..."}               回复文本 token（逐字追加）
    {"type": "tool_call",   "name": "...", "args": {...}} 工具开始调用（含参数）
    {"type": "tool_result", "name": "..."}                工具执行完毕
    {"type": "done"}                                       本轮回复结束
    {"type": "ping"}                                       保活帧（忽略）
    {"type": "error",       "message": "..."}              错误

─── SSE 消息协议 ────────────────────────────────────────────────────────
  每行格式：data: <JSON帧>\n\n
  JSON 结构与 WebSocket 帧相同，最后一帧 type=done。
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 工具事件 marker 前缀（由 agent.runner.AgentRunner.run_stream 发出）
_TOOL_CALL_PREFIX   = "\x00TOOL_CALL:"
_TOOL_RESULT_PREFIX = "\x00TOOL_RESULT:"

# WebSocket ping 间隔（秒）：防止工具执行期间（可能几十秒）连接被代理/浏览器超时关闭
_WS_PING_INTERVAL = 20

router = APIRouter()


# ── 内部：token → JSON 帧 ──────────────────────────────────────────────────────

def _parse_token(token: str) -> dict:
    """
    将 agent 内部 token 转换为 JSON 帧 dict。

    普通文本 → {"type": "token", "text": "..."}
    工具调用 → {"type": "tool_call",   "name": "...", "args": {...}}
    工具结果 → {"type": "tool_result", "name": "..."}

    marker 格式（agent.runner.AgentRunner.run_stream）：
      TOOL_CALL   前缀后跟 {"name":"...", "args":{...}} JSON
      TOOL_RESULT 前缀后跟 {"name":"..."} JSON
    """
    if token.startswith(_TOOL_CALL_PREFIX):
        payload_str = token[len(_TOOL_CALL_PREFIX):]
        try:
            payload = json.loads(payload_str)
            return {"type": "tool_call", "name": payload["name"], "args": payload.get("args", {})}
        except Exception:
            return {"type": "tool_call", "name": payload_str, "args": {}}

    if token.startswith(_TOOL_RESULT_PREFIX):
        payload_str = token[len(_TOOL_RESULT_PREFIX):]
        try:
            payload = json.loads(payload_str)
            return {"type": "tool_result", "name": payload["name"]}
        except Exception:
            return {"type": "tool_result", "name": payload_str}

    return {"type": "token", "text": token}


def _frame(data: dict) -> str:
    """序列化 JSON 帧为字符串（WebSocket send_text / SSE data 共用）。"""
    return json.dumps(data, ensure_ascii=False)


# ── 非流式 REST ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """单轮问答接口（REST，非流式，返回完整回复）。"""
    from gateway.server import get_app
    reply = await get_app().send(req.message)
    return ChatResponse(reply=reply)


# ── SSE 流式 REST ──────────────────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    SSE 流式问答接口（text/event-stream）。

    每条事件：data: <JSON帧>\\n\\n
    结束事件：data: {"type":"done"}\\n\\n
    """
    from gateway.server import get_app
    app = get_app()

    async def event_generator():
        try:
            async for token in app.send_stream(req.message):
                yield f"data: {_frame(_parse_token(token))}\n\n"
        except Exception as exc:
            yield f"data: {_frame({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            yield f"data: {_frame({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲，确保实时推送
        },
    )


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

    async def _stream_with_ping(text: str) -> None:
        """
        并发执行两个协程：
        - 主流：消费 app.send_stream()，将每个 token 转为 JSON 帧发送
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
            async for token in app.send_stream(text):
                await websocket.send_text(_frame(_parse_token(token)))
            await websocket.send_text(_frame({"type": "done"}))
        except Exception as exc:
            await websocket.send_text(_frame({"type": "error", "message": str(exc)}))
            await websocket.send_text(_frame({"type": "done"}))
        finally:
            done_event.set()
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

    try:
        while True:
            text = await websocket.receive_text()
            await _stream_with_ping(text)
    except WebSocketDisconnect:
        pass
