"""
tools/executor.py — 工具执行器

负责：
  - 参数 JSON 反序列化
  - schema-driven 参数 cast / 校验
  - 捕获可恢复执行错误
  - 超长结果 spill 到文件
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.registry import ToolRegistry


class ToolExecutor:
    """
    Tool 执行器。

    registry 负责“有哪些工具”，executor 负责“怎么执行一个工具”。
    """

    _RESULT_CHAR_LIMIT = 800 * 3

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        tool_call_dir: str | None = None,
    ):
        self._registry = registry
        self._tool_call_dir = Path(tool_call_dir) if tool_call_dir else Path.cwd()

    def _spill_to_file(self, tool_name: str, content: str) -> str:
        out_dir = self._tool_call_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = out_dir / f"{tool_name}_{ts}.txt"
        file_path.write_text(content, encoding="utf-8")

        return (
            f"[结果过长，已写入文件]\n"
            f"路径：{file_path}\n"
            f"共 {len(content)} 字符。\n"
            f"请使用 read_file 工具读取所需内容，例如：\n"
            f"  read_file(path='{file_path}')                        # 读取全文\n"
            f"  read_file(path='{file_path}', start_line=1, end_line=50)  # 读取前 50 行\n"
            f"  read_file(path='{file_path}', pattern='关键词')      # 按关键词过滤"
        )

    def _handle_result(self, tool_name: str, result: str) -> str:
        if len(result) <= self._RESULT_CHAR_LIMIT:
            return result
        return self._spill_to_file(tool_name, result)

    @staticmethod
    def _describe_tool_parameters(tool) -> tuple[list[str], list[str]]:
        schema = tool.parameters or {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        return list(properties.keys()), list(required)

    @staticmethod
    def _format_tool_arg_error(
        tool_name: str,
        message: str,
        allowed_names: list[str],
        required_names: list[str],
    ) -> str:
        parts = [f"[{tool_name} 参数错误] {message}"]
        if allowed_names:
            parts.append("允许参数：" + ", ".join(allowed_names))
        if required_names:
            parts.append("必填参数：" + ", ".join(required_names))
        parts.append("请根据工具 schema 修正参数后重试。")
        return "\n".join(parts)

    async def _call_tool(
        self,
        tool,
        args: dict[str, Any],
    ) -> str:
        allowed_names, required_names = self._describe_tool_parameters(tool)
        args = tool.cast_params(args)
        errors = tool.validate_params(args)
        if errors:
            return self._format_tool_arg_error(
                tool_name=tool.name,
                message="；".join(errors),
                allowed_names=allowed_names,
                required_names=required_names,
            )

        try:
            result = await tool.execute(**args)
        except Exception as exc:
            return f"[{tool.name} 失败] {exc}"

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, indent=2)

        if not tool.allow_spill:
            return result
        return self._handle_result(tool.name, result)

    async def run(self, name: str, arguments: str) -> str:
        try:
            args: dict[str, Any] = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        tool = self._registry.get(name)
        if tool is None:
            return f"[未知工具: {name}]"

        return await self._call_tool(tool=tool, args=args)
