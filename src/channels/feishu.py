"""
channels/feishu.py — 飞书 / Lark 事件订阅 webhook

接收飞书推送的消息事件，通过 get_app().send() 进入 AIButlerApp.inbox 队列，
Agent 主循环处理后将回复通过飞书 OpenAPI 发回用户。

接入文档：https://open.feishu.cn/document/ukTMukTMukTM/uUTNz4SN1MjL1UzM
"""
from fastapi import APIRouter, Request

from gateway.server import get_app

router = APIRouter()


@router.post("/event")
async def feishu_event(request: Request):
    """接收飞书事件订阅推送。"""
    payload = await request.json()

    # 飞书 URL 验证握手
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    # TODO: 验证签名（X-Lark-Signature）
    # TODO: 解析 payload，提取消息文本和 open_chat_id
    # TODO: 调用 get_app().send(text) 获取回复
    # TODO: 通过飞书 OpenAPI 发送回复消息

    return {"msg": "ok"}
