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

    working_dir: str = "data/.reme"
    max_input_length: int = 128000
    compact_ratio: float = 0.7
    memory_compact_reserve: int = 10000

    # 静默检索：相似度 >= 此阈值才把历史记忆注入 system
    # 范围 0~1，越高越严格（减少无关噪声注入）
    # ReMe hybrid search 分数 = vector_score*0.7 + keyword_score*0.3，
    # 典型有效匹配在 0.5~0.8 之间，0.75 过高会漏掉大量有效召回
    memory_similarity_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "glm-5"),
            emb_base_url=os.getenv("EMB_BASE_URL", ""),
            emb_api_key=os.getenv("EMB_API_KEY", ""),
            emb_model=os.getenv("EMB_MODEL", ""),
            memory_similarity_threshold=float(
                os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.75")
            ),
        )
