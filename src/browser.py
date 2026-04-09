"""
Headless 浏览器操作模块（基于 browser-use）

browser-use 是专为 AI Agent 设计的浏览器自动化库，
Agent 只需用自然语言描述任务，browser-use 的内置 Agent 会自主完成：
  - 页面导航、元素定位、点击、输入
  - 截图、内容提取
  - 多步骤复杂任务（如填表、搜索比价等）

设计要点：
  - 懒初始化：首次调用时才 import browser-use 和启动浏览器
  - 会话保持：同一轮对话共享 Browser 实例
  - 复用 AI Butler 已有的 LLM 配置（通过 OpenAI 兼容接口）
  - 自动清理：提供 close 方法，main.py 退出时调用
"""
import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class BrowserUseConfig:
    """browser-use 配置"""
    # 是否 headless 模式
    headless: bool = True
    # browser-use Agent 最大操作步数（防止无限循环）
    max_steps: int = 20
    # 浏览器使用的 LLM（复用 AI Butler 的 LLM 配置）
    # 格式: openai 兼容，通过 ChatOpenAI 传入
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""


class BrowserAgent:
    """
    browser-use 浏览器 Agent 封装。

    作为 AI Butler 的工具，接收自然语言任务描述，
    由 browser-use 的 Agent 自主操作浏览器完成。
    """

    def __init__(self, config: BrowserUseConfig | None = None):
        self.config = config or BrowserUseConfig()
        self._browser = None      # browser_use.Browser 实例
        self._initialized = False

    async def _ensure_initialized(self):
        """懒初始化：首次操作时才导入 browser-use 并启动浏览器"""
        if self._initialized:
            return

        try:
            from browser_use import Browser, BrowserConfig
        except ImportError:
            raise RuntimeError(
                "browser-use 未安装。请运行:\n"
                "  pip install browser-use\n"
                "  playwright install chromium"
            )

        browser_config = BrowserConfig(
            headless=self.config.headless,
        )
        self._browser = Browser(config=browser_config)
        self._initialized = True

    def _create_llm(self):
        """创建供 browser-use Agent 使用的 LLM 实例（OpenAI 兼容）"""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
        )

    async def run_task(self, task: str) -> dict:
        """
        执行浏览器任务。

        Args:
            task: 自然语言任务描述，如：
                  - "打开 https://example.com 并提取页面标题"
                  - "在百度搜索 Python 教程，返回前 5 条结果"
                  - "打开 GitHub trending 页面，截图保存"

        Returns:
            {
                "success": bool,
                "result": str,       # Agent 的最终回答/操作结果
                "steps": int,        # 执行步数
                "error": str | None,
            }
        """
        await self._ensure_initialized()

        try:
            from browser_use import Agent
        except ImportError:
            return {
                "success": False,
                "result": "",
                "steps": 0,
                "error": "browser-use 未安装",
            }

        llm = self._create_llm()

        try:
            agent = Agent(
                task=task,
                llm=llm,
                browser=self._browser,
                max_steps=self.config.max_steps,
            )

            result = await agent.run()

            # browser-use Agent.run() 返回 AgentHistoryList
            # 从中提取最终结果
            final_result = result.final_result() if result else ""
            steps_count = len(result.history) if result and hasattr(result, 'history') else 0

            return {
                "success": True,
                "result": final_result or "任务已完成，但未产生文本结果。",
                "steps": steps_count,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "steps": 0,
                "error": str(e),
            }

    async def close(self):
        """关闭浏览器并释放资源"""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
