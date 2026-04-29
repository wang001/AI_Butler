"""
provider/factory.py — Provider 构造入口
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from provider.base import ChatProvider
from provider.openai_compatible_provider import OpenAICompatibleProvider
from provider.openai_responses_provider import OpenAIResponsesProvider

if TYPE_CHECKING:
    from config import Config


def build_chat_provider(
    cfg: "Config",
    *,
    model: str | None = None,
) -> ChatProvider:
    provider_name = (getattr(cfg, "llm_provider", "auto") or "auto").strip().lower()
    effective_model = model or cfg.llm_model

    if provider_name in ("openai_responses", "openai-responses", "responses"):
        return OpenAIResponsesProvider(
            model=effective_model,
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
        )

    if provider_name in ("", "auto", "openai", "openai_compatible", "openai-compatible"):
        return OpenAICompatibleProvider(
            model=effective_model,
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
        )

    raise ValueError(f"unsupported llm provider: {provider_name}")
