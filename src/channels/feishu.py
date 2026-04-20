"""
channels/feishu.py — 飞书 / Lark 事件订阅 Channel

接收飞书推送的消息事件，通过 get_app().send() 进入 AIButlerApp.inbox 队列，
Agent 主循环处理后将回复通过飞书 OpenAPI 发回用户。

支持的消息类型：
  - 文本消息：直接透传给 Agent
  - 图片消息：下载图片 → 构造描述提示词 → 传给 Agent（需 LLM 支持 vision）
  - 语音消息：下载语音 → 语音转文字提示 → 传给 Agent
  - 文件消息：下载文件 → 文本提取 → 传给 Agent

接入文档：https://open.feishu.cn/document/ukTMukTMukTM/uUTNz4SN1MjL1UzM
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

from fastapi import APIRouter, BackgroundTasks, Request, Response

from gateway.server import get_app

logger = logging.getLogger("feishu.channel")

router = APIRouter()

# ── 飞书 API 客户端（懒加载单例）───────────────────────────────────────────────

_api_client: "FeishuAPI | None" = None


def _get_api() -> "FeishuAPI":
    """获取飞书 API 客户端单例。"""
    global _api_client
    if _api_client is None:
        from channels.feishu_api import FeishuAPI
        from config import Config
        cfg = Config.from_env()
        if not cfg.feishu_app_id or not cfg.feishu_app_secret:
            raise RuntimeError("飞书配置缺失：请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        _api_client = FeishuAPI(cfg.feishu_app_id, cfg.feishu_app_secret)
    return _api_client


# ── 签名校验 ─────────────────────────────────────────────────────────────────────

def _verify_signature(timestamp: str, nonce: str, body: bytes, sign_secret: str, signature: str) -> bool:
    """
    验证飞书事件推送签名。

    算法：signature = base64(hmac_sha256(EncryptKey, timestamp + nonce + body))
    文档：https://open.feishu.cn/document/ukTMukTMukTM/uYDNxYjL2QTM24iN0EjN/event-subscription-verify
    """
    if not sign_secret:
        logger.warning("FEISHU_SIGN_SECRET 未配置，跳过签名校验")
        return True

    string_to_sign = f"{timestamp}{nonce}" + body.decode("utf-8", errors="replace")
    hmac_obj = hmac.new(
        sign_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    )
    calculated_signature = hmac_obj.hexdigest()
    return hmac.compare_digest(calculated_signature, signature)


# ── 事件处理 ─────────────────────────────────────────────────────────────────────

async def _handle_message_event(event: dict, header: dict) -> None:
    """
    处理飞书消息事件。

    流程：解析消息 → 构造 Agent 输入 → send() 获取回复 → 通过飞书 API 回复
    """
    msg_type = event.get("message_type", "text")
    message_id = event.get("message_id", "")
    chat_id = event.get("chat_id", "")
    sender = event.get("sender", {})
    sender_id = sender.get("sender_id", {})
    open_id = sender_id.get("open_id", "")
    is_group = event.get("chat_type") == "group"

    api = _get_api()

    # ── 根据 msg_type 提取用户输入 ──────────────────────────────────────────
    agent_input = await _extract_message_content(event, api)

    if not agent_input:
        logger.warning("无法提取消息内容，跳过: msg_type=%s, message_id=%s", msg_type, message_id)
        return

    # ── 群聊 @机器人 过滤 ───────────────────────────────────────────────────
    if is_group:
        mentions = event.get("mentions", [])
        # 如果群聊中没有人 @机器人，则忽略
        bot_name = os.getenv("FEISHU_BOT_NAME", "")
        mentioned = False
        for m in mentions:
            if m.get("name") == bot_name or m.get("id", {}).get("open_id") == open_id:
                mentioned = True
                # 从文本中移除 @机器人 的 mention key
                mention_key = m.get("key", "")
                if mention_key:
                    agent_input = agent_input.replace(f"@{m.get('name', '')}", "").replace(mention_key, "")
                break
        if not mentioned and bot_name:
            logger.debug("群聊消息未 @机器人，忽略: chat_id=%s", chat_id)
            return

    agent_input = agent_input.strip()
    if not agent_input:
        return

    logger.info("收到飞书消息: chat_id=%s, msg_type=%s, len=%d", chat_id, msg_type, len(agent_input))

    # ── 调用 Agent ──────────────────────────────────────────────────────────
    try:
        app = get_app()
        reply = await app.send(agent_input)
    except Exception as e:
        logger.error("Agent 处理失败: %s", e)
        reply = f"抱歉，处理消息时出错：{e}"

    # ── 回复飞书 ────────────────────────────────────────────────────────────
    if not reply:
        reply = "（思考中...但没有产出回复）"

    try:
        # 优先使用 reply（引用原消息），回退到 send_text
        result = await api.reply_text(message_id, reply)
        if result.get("code") != 0:
            # reply 失败时回退到主动发送
            logger.warning("reply 失败 (%s)，尝试主动发送到 chat_id=%s", result.get("msg"), chat_id)
            await api.send_text(chat_id, reply, receive_id_type="chat_id")
    except Exception as e:
        logger.error("飞书回复失败: %s", e)


async def _extract_message_content(event: dict, api: "FeishuAPI") -> str:
    """
    根据消息类型提取用户输入文本。

    - text: 直接解析 content JSON
    - image: 下载图片 → 提示 LLM 识别
    - audio: 下载语音 → 提示 LLM 转写
    - file:  下载文件 → 提取文本内容
    - post:  富文本 → 提取纯文本
    """
    import json as _json

    msg_type = event.get("message_type", "text")
    content_str = event.get("message", {}).get("content", "{}")
    message_id = event.get("message_id", "")

    try:
        content = _json.loads(content_str)
    except Exception:
        content = {}

    if msg_type == "text":
        return content.get("text", "")

    elif msg_type == "post":
        # 富文本：提取所有文本段落
        return _extract_post_text(content)

    elif msg_type == "image":
        image_key = content.get("image_key", "")
        if not image_key:
            return "[收到一张图片，但无法获取图片 key]"
        # 下载图片并构造描述提示
        try:
            image_data = await api.download_resource(message_id, image_key, "image")
            # 当前 Agent 仅支持文本输入，将图片信息作为提示传入
            # TODO: 当 Butler 支持 vision 时，传入图片数据
            size_kb = len(image_data) / 1024
            return f"[用户发送了一张图片，图片 key: {image_key}，大小: {size_kb:.1f}KB]\n请描述一下你看到了什么？"
        except Exception as e:
            logger.error("下载图片失败: %s", e)
            return f"[收到图片但下载失败: {e}]"

    elif msg_type == "audio":
        file_key = content.get("file_key", "")
        if not file_key:
            return "[收到一条语音，但无法获取文件 key]"
        try:
            audio_data = await api.download_resource(message_id, file_key, "audio")
            duration = content.get("duration", 0)
            # 当前 Agent 仅支持文本输入，将语音信息作为提示传入
            # TODO: 接入语音转文字服务（如 Whisper）
            size_kb = len(audio_data) / 1024
            return f"[用户发送了一条语音，时长: {duration}ms，大小: {size_kb:.1f}KB]\n请将这段语音转写为文字。"
        except Exception as e:
            logger.error("下载语音失败: %s", e)
            return f"[收到语音但下载失败: {e}]"

    elif msg_type == "file":
        file_key = content.get("file_key", "")
        file_name = content.get("file_name", "unknown")
        if not file_key:
            return f"[收到文件 {file_name}，但无法获取文件 key]"
        try:
            file_data = await api.download_resource(message_id, file_key, "file")
            size_kb = len(file_data) / 1024
            # 尝试提取文本内容
            text_content = _try_extract_file_text(file_data, file_name)
            if text_content:
                return f"[用户发送了文件: {file_name}，大小: {size_kb:.1f}KB]\n文件内容:\n{text_content}"
            else:
                return f"[用户发送了文件: {file_name}，大小: {size_kb:.1f}KB，无法提取文本内容]"
        except Exception as e:
            logger.error("下载文件失败: %s", e)
            return f"[收到文件 {file_name} 但下载失败: {e}]"

    else:
        return f"[收到一条 {msg_type} 类型的消息，暂不支持处理]"


def _extract_post_text(content: dict) -> str:
    """从飞书富文本消息中提取纯文本。"""
    parts: list[str] = []
    # 富文本结构：{"zh_cn": {"title": "...", "content": [[{...}, ...], ...]}}
    for lang_key in ("zh_cn", "en_us", "ja_jp"):
        lang_content = content.get(lang_key)
        if not lang_content:
            continue
        title = lang_content.get("title", "")
        if title:
            parts.append(title)
        for line in lang_content.get("content", []):
            for element in line:
                tag = element.get("tag", "")
                if tag == "text":
                    parts.append(element.get("text", ""))
                elif tag == "a":
                    parts.append(element.get("text", element.get("href", "")))
                elif tag == "at":
                    parts.append(f"@{element.get('user_name', element.get('user_id', ''))}")
        break  # 只处理第一个语言版本
    return "".join(parts)


def _try_extract_file_text(file_data: bytes, filename: str) -> str:
    """
    尝试从文件中提取文本内容。

    支持：纯文本、JSON、CSV、Markdown 等文本文件。
    不支持：PDF、Word、Excel 等二进制格式（需要额外库）。
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # 文本类文件直接解码
    text_extensions = {"txt", "md", "csv", "json", "xml", "yaml", "yml", "toml", "ini", "log", "py", "js", "ts", "java", "go", "rs", "c", "cpp", "h", "sh", "bash", "sql", "html", "css", "conf", "env"}
    if ext in text_extensions:
        try:
            return file_data.decode("utf-8", errors="replace")[:50000]  # 限制长度
        except Exception:
            pass

    # 其他格式暂不支持
    return ""


# ── 路由 ─────────────────────────────────────────────────────────────────────────

@router.post("/event")
async def feishu_event(request: Request, background_tasks: BackgroundTasks):
    """
    接收飞书事件订阅推送。

    飞书事件推送为 HTTP POST JSON，包含 header + event。
    重要：飞书要求在 3 秒内返回 200，因此将耗时操作放入后台任务。
    """
    body = await request.body()

    # ── 飞书 URL 验证握手 ────────────────────────────────────────────────────
    try:
        payload = json.loads(body)
    except Exception:
        return Response(content="invalid json", status_code=400)

    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge", "")
        logger.info("飞书 URL 验证握手: challenge=%s", challenge)
        return {"challenge": challenge}

    # ── 签名校验 ────────────────────────────────────────────────────────────
    header = payload.get("header", {})
    timestamp = header.get("timestamp", "")
    nonce = header.get("nonce", "")
    signature = header.get("signature", "")

    from config import Config
    cfg = Config.from_env()
    if cfg.feishu_sign_secret:
        if not _verify_signature(timestamp, nonce, body, cfg.feishu_sign_secret, signature):
            logger.warning("飞书签名校验失败")
            return Response(content="signature mismatch", status_code=403)

    # ── 事件处理（后台任务，避免超时） ────────────────────────────────────────
    event_type = header.get("event_type", "")
    event = payload.get("event", {})

    if event_type == "im.message.receive_v1":
        background_tasks.add_task(_handle_message_event, event, header)
    else:
        logger.debug("忽略非消息事件: event_type=%s", event_type)

    # 飞书要求 3 秒内返回 200
    return {"code": 0, "msg": "ok"}


# ── 健康检查 / 测试 ─────────────────────────────────────────────────────────────

@router.get("/test")
async def feishu_test():
    """飞书 Channel 健康检查。"""
    from config import Config
    cfg = Config.from_env()
    configured = bool(cfg.feishu_app_id and cfg.feishu_app_secret)
    return {
        "channel": "feishu",
        "configured": configured,
        "app_id_set": bool(cfg.feishu_app_id),
        "sign_secret_set": bool(cfg.feishu_sign_secret),
    }
