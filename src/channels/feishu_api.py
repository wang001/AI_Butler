"""
channels/feishu_api.py — 飞书 / Lark OpenAPI 客户端

封装飞书开放平台 API 调用，包括：
  - tenant_access_token 获取与自动刷新
  - 消息发送（文本 / 富文本 / 图片 / 文件）
  - 文件 / 图片下载
  - 文件 / 图片上传

文档：https://open.feishu.cn/document/server-docs/getting-started/api-access-token/tenant-access-token
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

logger = logging.getLogger("feishu.api")

_FEISHU_BASE = "https://open.feishu.cn"


class FeishuAPI:
    """飞书 OpenAPI 异步客户端。"""

    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self._token: str = ""
        self._token_expires: float = 0.0
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                base_url=_FEISHU_BASE,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Token 管理 ───────────────────────────────────────────────────────────────

    async def _ensure_token(self) -> str:
        """获取有效的 tenant_access_token，自动刷新。"""
        async with self._lock:
            if self._token and time.time() < self._token_expires - 60:
                return self._token
            session = await self._get_session()
            resp = await session.post(
                "/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            data = await resp.json()
            if data.get("code") != 0:
                logger.error("获取 tenant_access_token 失败: %s", data)
                raise RuntimeError(f"获取 tenant_access_token 失败: {data.get('msg')}")
            self._token = data["tenant_access_token"]
            self._token_expires = time.time() + data.get("expire", 7200)
            logger.info("tenant_access_token 已刷新，有效期 %ds", data.get("expire", 7200))
            return self._token

    # ── 通用请求 ─────────────────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        data: Any = None,
    ) -> dict:
        """发送 API 请求，自动附带 Authorization 头。"""
        token = await self._ensure_token()
        session = await self._get_session()
        hdrs = {"Authorization": f"Bearer {token}"}
        if headers:
            hdrs.update(headers)
        if data is not None and isinstance(data, aiohttp.FormData):
            hdrs.pop("Content-Type", None)
        async with session.request(
            method, path, json=json_data, params=params, headers=hdrs, data=data,
        ) as resp:
            result = await resp.json()
        if result.get("code") != 0:
            logger.warning("飞书 API 返回错误: %s %s → %s", method, path, result.get("msg"))
        return result

    # ── 消息发送 ─────────────────────────────────────────────────────────────────

    async def send_message(
        self,
        receive_id: str,
        msg_type: str,
        content: str,
        *,
        receive_id_type: str = "chat_id",
    ) -> dict:
        """
        发送消息（通用接口）。

        Args:
            receive_id: 接收者 ID
            msg_type: 消息类型 (text / post / image / file / audio)
            content: 消息内容 JSON 字符串
            receive_id_type: receive_id 类型 (chat_id / open_id / user_id)
        """
        return await self._request(
            "POST",
            "/open-apis/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            json_data={
                "receive_id": receive_id,
                "msg_type": msg_type,
                "content": content,
            },
        )

    async def send_text(self, receive_id: str, text: str, **kw) -> dict:
        """发送纯文本消息。"""
        import json
        content = json.dumps({"text": text})
        return await self.send_message(receive_id, "text", content, **kw)

    async def send_post(
        self, receive_id: str, title: str, content: list[list[dict]], **kw
    ) -> dict:
        """发送富文本消息。"""
        import json
        body = json.dumps({"zh_cn": {"title": title, "content": content}})
        return await self.send_message(receive_id, "post", body, **kw)

    async def send_image(self, receive_id: str, image_key: str, **kw) -> dict:
        """发送图片消息。"""
        import json
        content = json.dumps({"image_key": image_key})
        return await self.send_message(receive_id, "image", content, **kw)

    async def send_file(self, receive_id: str, file_key: str, **kw) -> dict:
        """发送文件消息。"""
        import json
        content = json.dumps({"file_key": file_key})
        return await self.send_message(receive_id, "file", content, **kw)

    async def reply_message(self, message_id: str, msg_type: str, content: str) -> dict:
        """回复消息（引用原消息）。"""
        return await self._request(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reply",
            json_data={"msg_type": msg_type, "content": content},
        )

    async def reply_text(self, message_id: str, text: str) -> dict:
        """回复文本消息。"""
        import json
        content = json.dumps({"text": text})
        return await self.reply_message(message_id, "text", content)

    # ── 消息撤回 / 更新 ─────────────────────────────────────────────────────────

    async def update_message(self, message_id: str, content: str) -> dict:
        """更新消息内容（仅文本/富文本支持）。"""
        return await self._request(
            "PATCH",
            f"/open-apis/im/v1/messages/{message_id}",
            json_data={"content": content},
        )

    # ── 资源下载 ─────────────────────────────────────────────────────────────────

    async def download_resource(
        self, message_id: str, file_key: str, resource_type: str = "file"
    ) -> bytes:
        """
        下载消息中的文件/图片/语音资源。

        Args:
            message_id: 消息 ID
            file_key: 文件 key
            resource_type: 资源类型 (file / image / audio)
        """
        token = await self._ensure_token()
        session = await self._get_session()
        params = {"type": resource_type}
        headers = {"Authorization": f"Bearer {token}"}
        async with session.get(
            f"/open-apis/im/v1/messages/{message_id}/resources/{file_key}",
            params=params,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error("下载资源失败: %s", text)
                raise RuntimeError(f"下载资源失败: {resp.status}")
            return await resp.read()

    # ── 文件 / 图片上传 ─────────────────────────────────────────────────────────

    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
        file_type: str = "stream",
        *,
        parent_type: str = "im_message",
        parent_id: str = "",
    ) -> dict:
        """
        上传文件，获取 file_key。

        Args:
            file_data: 文件二进制数据
            filename: 文件名
            file_type: 文件类型 (stream / pdf / doc / xls / ppt / stream)
            parent_type: 父对象类型 (im_message / im_message_resource)
            parent_id: 父对象 ID（发送消息时留空即可）
        """
        form = aiohttp.FormData()
        form.add_field("file_type", file_type)
        form.add_field("file_name", filename)
        if parent_id:
            form.add_field("parent_id", parent_id)
        else:
            form.add_field("parent_id", " ", )  # 飞书 API 要求必填，无实际 parent 时传空格
        form.add_field("parent_type", parent_type)
        form.add_field("file", file_data, filename=filename, content_type="application/octet-stream")
        return await self._request(
            "POST", "/open-apis/im/v1/files", data=form,
        )

    async def upload_image(self, image_data: bytes, image_type: str = "message") -> dict:
        """
        上传图片，获取 image_key。

        Args:
            image_data: 图片二进制数据
            image_type: 图片类型 (message / avatar)
        """
        form = aiohttp.FormData()
        form.add_field("image_type", image_type)
        form.add_field("image", image_data, filename="image.png", content_type="image/png")
        return await self._request(
            "POST", "/open-apis/im/v1/images", data=form,
        )
