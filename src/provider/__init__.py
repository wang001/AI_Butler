"""
provider — 模型 Provider 层

职责：
  - 封装不同模型/兼容层的消息 role 兼容性
  - 封装非流式 / 流式补全请求
  - 统一提取 reasoning / text delta
"""
from provider.base import (
    ChatProvider,
    ContentBlockFilter,
    ProviderModelConfig,
    ProviderToolCall,
    ProviderToolCallDelta,
)
from provider.factory import build_chat_provider
from provider.openai_compatible_provider import OpenAICompatibleProvider
from provider.openai_responses_provider import OpenAIResponsesProvider

__all__ = [
    "ChatProvider",
    "ContentBlockFilter",
    "ProviderModelConfig",
    "ProviderToolCall",
    "ProviderToolCallDelta",
    "OpenAICompatibleProvider",
    "OpenAIResponsesProvider",
    "build_chat_provider",
]
