import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "glm-5"

    emb_base_url: str = ""
    emb_api_key: str = ""
    emb_model: str = ""

    working_dir: str = "./data/memory"
    max_input_length: int = 128000
    compact_ratio: float = 0.7
    memory_compact_reserve: int = 10000

    # 静默检索：相似度 >= 此阈值才把历史记忆注入 system
    # 范围 0~1，越高越严格（减少无关噪声注入）
    # ReMe hybrid search 分数 = vector_score*0.7 + keyword_score*0.3，
    # 典型有效匹配在 0.5~0.8 之间，0.75 过高会漏掉大量有效召回
    memory_similarity_threshold: float = 0.5

    # ── 命令执行配置（容器内直接 subprocess 执行） ──
    command_enabled: bool = True
    command_workdir: str = "/workspace"
    command_default_timeout: int = 30

    # ── browser-use 配置 ──
    browser_enabled: bool = True
    browser_headless: bool = True
    browser_max_steps: int = 20

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "glm-5"),
            emb_base_url=os.getenv("EMB_BASE_URL", ""),
            emb_api_key=os.getenv("EMB_API_KEY", ""),
            emb_model=os.getenv("EMB_MODEL", ""),
            working_dir=os.getenv("WORKING_DIR", "/data/memory"),
            memory_similarity_threshold=float(
                os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.75")
            ),
            command_enabled=os.getenv("COMMAND_ENABLED", "true").lower() == "true",
            command_workdir=os.getenv("COMMAND_WORKDIR", "/workspace"),
            command_default_timeout=int(os.getenv("COMMAND_DEFAULT_TIMEOUT", "30")),
            browser_enabled=os.getenv("BROWSER_ENABLED", "true").lower() == "true",
            browser_headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true",
            browser_max_steps=int(os.getenv("BROWSER_MAX_STEPS", "20")),
        )
