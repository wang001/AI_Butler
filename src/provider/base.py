"""
provider/base.py — 模型 Provider 抽象

目标：
  - 将不同模型/兼容层的消息 role、流式 delta、reasoning 字段差异
    从 AgentRunner 中抽离
  - Runner 只描述“对话循环”，Provider 负责“怎么和模型说话”
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderModelConfig:
    """Provider 导出的旁路模型连接配置。"""

    model: str
    api_key: str = ""
    base_url: str = ""
    litellm_provider: str = "openai"

    @property
    def litellm_model(self) -> str:
        prefix = f"{self.litellm_provider}/"
        return self.model if self.model.startswith(prefix) else f"{prefix}{self.model}"


@dataclass(frozen=True)
class ProviderToolCallDelta:
    """Provider 流式返回中的一段 tool call 增量。"""

    index: int
    id: str = ""
    name: str = ""
    arguments_delta: str = ""


@dataclass(frozen=True)
class ProviderToolCall:
    """Runner 内部使用的完整 tool call。"""

    id: str
    name: str
    arguments: str = "{}"


class ContentBlockFilter(Protocol):
    """Provider 级内容过滤器：逐 chunk 拆分 reasoning / text。"""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        ...

    def flush(self) -> list[tuple[str, str]]:
        ...


class ChatProvider(ABC):
    def __init__(self, *, model: str):
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 标识。"""

    @abstractmethod
    async def create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        stream: bool = False,
    ) -> Any:
        """创建一次聊天补全。返回 provider 原生 response / stream。"""

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """在发给 provider 前重写消息，例如兼容不支持 system role 的模型。"""
        return messages

    @abstractmethod
    def split_reasoning_and_reply(self, text: str) -> tuple[str, str]:
        """将非流式完整文本拆分成 reasoning 与最终展示文本。"""

    @abstractmethod
    def extract_delta_reasoning(self, delta: Any) -> str:
        """从 provider 的流式 delta 中提取 reasoning 文本。"""

    @abstractmethod
    def extract_delta_content(self, delta: Any) -> str:
        """从 provider 的流式 delta 中提取 content 文本。"""

    @abstractmethod
    def create_content_block_filter(self) -> ContentBlockFilter:
        """创建逐 chunk 的 content 过滤器。"""

    @abstractmethod
    def extract_delta_tool_calls(self, delta: Any) -> list[ProviderToolCallDelta]:
        """从 provider 的流式 delta 中提取 tool call 参数增量。"""

    @abstractmethod
    def export_model_config(self) -> ProviderModelConfig:
        """导出给 ReMe、browser-use、subAgent 等旁路组件复用的模型配置。"""
