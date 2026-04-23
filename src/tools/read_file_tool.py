"""
tools/read_file_tool.py — 纯文本文件读取工具

【仅支持可按指定编码解码的文本文件】：.txt / .md / .log / .json / .yaml / .csv /
.py / .sh 等。若文件以指定编码解码失败，会提示尝试其他编码或改用 run_command 处理。

供模型通过 Tool Call 读取文本文件内容，主要用于：
  1. 读取 ToolDispatcher 溢出写入的大结果文件（/data/tool_call/）
  2. 读取工作目录下的配置、日志、数据等文本文件

支持五种读取模式（参数自由组合）：
  - 全文读取   : 只传 path
  - 行范围读取 : 传 start_line / end_line，适合已知结构的大文件分段读取
  - 关键词过滤 : 传 pattern（正则），只返回匹配行
  - 搜索+上下文: pattern + context_lines，类似 grep -C
  - 字符截断   : 传 max_chars，对最终返回文本做字符数上限控制

注意：此工具的返回结果不受大小限制写入临时文件，自身有 _MAX_LINES 行数兜底。
"""
import re
from pathlib import Path
from typing import Optional

from tools.base import Tool

# 常见纯文本扩展名（仅用于友好提示，不作强制拦截）
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".log", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env",
    ".csv", ".tsv",
    ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".sh", ".bash", ".zsh", ".fish",
    ".html", ".htm", ".xml", ".css",
    ".sql", ".graphql",
}

# ── OpenAI function-calling schema ────────────────────────────────────────────

FILE_READER_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "读取本地【纯文本】文件内容。\n"
                "仅支持可用文本编辑器打开的文件：.txt / .md / .log / .json / .yaml / "
                ".csv / .py / .sh / .html 等。\n"
                "⚠ 不支持二进制文件（图片、PDF、Word、Excel、zip 等），"
                "如需处理二进制文件请使用 run_command。\n\n"
                "支持四种读取模式（可组合）：\n"
                "  1. 全文读取：只传 path\n"
                "  2. 行范围：传 start_line / end_line 读取指定行区间\n"
                "  3. 关键词过滤：传 pattern（正则），只返回匹配行\n"
                "  4. 搜索+上下文：pattern + context_lines，看匹配行的前后几行\n"
                "  5. 字符截断：传 max_chars，对最终输出做字符上限控制\n\n"
                "典型用法：\n"
                "  - 工具结果过长被写入文件后，用此工具按需读取\n"
                "  - 读取工作目录下的配置、日志、数据文件"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件的绝对路径或相对路径",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "起始行号（1-based，含），不传则从第 1 行开始",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "结束行号（1-based，含），不传则读到文件末尾",
                    },
                    "pattern": {
                        "type": "string",
                        "description": (
                            "正则表达式或普通关键词，只返回匹配的行。"
                            "可与 start_line/end_line 组合，在行范围内过滤。"
                        ),
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": (
                            "匹配行前后各保留的行数（类似 grep -C），"
                            "仅在传入 pattern 时生效，默认 0"
                        ),
                        "default": 0,
                    },
                    "encoding": {
                        "type": "string",
                        "description": "文件编码，默认 utf-8",
                        "default": "utf-8",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": (
                            "最终返回文本的最大字符数。"
                            "适合在已知文件较大时先做粗截断，默认不限制。"
                        ),
                        "minimum": 200,
                        "maximum": 50000,
                    },
                },
                "required": ["path"],
            },
        },
    }
]

# ── 工具实现 ───────────────────────────────────────────────────────────────────

# 单次返回的最大行数，防止模型一次读取过多
_MAX_LINES = 500


def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    pattern: Optional[str] = None,
    context_lines: int = 0,
    encoding: str = "utf-8",
    max_chars: Optional[int] = None,
) -> str:
    """
    读取文件内容，返回字符串结果。

    Args:
        path          : 文件路径
        start_line    : 起始行（1-based，含），None 表示从头
        end_line      : 结束行（1-based，含），None 表示到尾
        pattern       : 正则过滤，只返回匹配行（及 context_lines 行上下文）
        context_lines : pattern 匹配时，前后各保留的行数
        encoding      : 文件编码，默认 utf-8
        max_chars     : 最终返回文本最大字符数，None 表示不限制

    Returns:
        文件内容字符串，包含行号前缀，格式：
            L10 | content
    """
    p = Path(path)

    if not p.exists():
        return f"[read_file 错误] 文件不存在：{path}"
    if not p.is_file():
        return f"[read_file 错误] 路径不是文件：{path}"

    # ── 文本解码（解码失败即视为非文本文件）────────────────────────────────
    try:
        raw_lines = p.read_text(encoding=encoding).splitlines()
    except UnicodeDecodeError:
        alt = "gbk" if encoding == "utf-8" else "utf-8"
        return (
            f"[read_file 错误] 文件无法以 {encoding} 解码，可能是二进制文件或其他编码。\n"
            f"文件：{path}\n"
            f"建议：\n"
            f"  1. 尝试其他编码，如 encoding=\"{alt}\" 或 encoding=\"utf-16\"\n"
            f"  2. 用 run_command 检测编码：file -i '{path}'\n"
            f"  3. 若确为二进制文件（图片/PDF/zip 等），请用 run_command 处理"
        )
    except Exception as e:
        return f"[read_file 错误] 读取失败：{e}"

    total = len(raw_lines)

    # ── Step 1: 行范围截取 ──────────────────────────────────────────────────
    s = max(1, start_line or 1)
    e = min(total, end_line or total)
    if s > total:
        return f"[read_file] 文件共 {total} 行，start_line={s} 超出范围。"

    # 取目标行，保留 0-based 索引供后续对齐行号
    selected: list[tuple[int, str]] = [
        (i + 1, raw_lines[i]) for i in range(s - 1, e)
    ]

    # ── Step 2: 正则过滤 + 上下文 ───────────────────────────────────────────
    if pattern:
        try:
            regex = re.compile(pattern)
        except re.error as ex:
            return f"[read_file 错误] 正则表达式无效：{ex}"

        # 找出匹配行的 0-based 索引（相对于 selected 列表）
        hit_indices: set[int] = {
            idx for idx, (_, line) in enumerate(selected) if regex.search(line)
        }

        if not hit_indices:
            scope = f"第 {s}–{e} 行" if (start_line or end_line) else "全文"
            return f"[read_file] 在{scope}中未找到匹配 `{pattern}` 的行。"

        # 展开上下文
        keep: set[int] = set()
        n = len(selected)
        for idx in hit_indices:
            for offset in range(-context_lines, context_lines + 1):
                ci = idx + offset
                if 0 <= ci < n:
                    keep.add(ci)

        selected = [selected[i] for i in sorted(keep)]

    # ── Step 3: 行数限制 ────────────────────────────────────────────────────
    truncated = False
    if len(selected) > _MAX_LINES:
        selected = selected[:_MAX_LINES]
        truncated = True

    # ── Step 4: 格式化输出 ──────────────────────────────────────────────────
    width = len(str(total))
    lines_out = [f"L{lineno:0{width}d} | {content}" for lineno, content in selected]
    output = "\n".join(lines_out)

    # 附加元信息头
    header_parts = [f"文件：{p}", f"共 {total} 行"]
    if start_line or end_line:
        header_parts.append(f"读取范围：第 {s}–{e} 行")
    if pattern:
        header_parts.append(f"过滤模式：{pattern}")
    if truncated:
        header_parts.append(f"⚠ 结果超过 {_MAX_LINES} 行，已截断，请缩小范围或使用 pattern 过滤")

    header = "  |  ".join(header_parts)
    result = f"[{header}]\n{output}"

    if max_chars is not None:
        max_chars = max(200, min(max_chars, 50000))
        if len(result) > max_chars:
            note = f"\n\n[read_file] 已按 max_chars={max_chars} 截断输出，请缩小范围或使用 pattern 过滤。"
            header_text = f"[{header}]"
            body_budget = max_chars - len(header_text) - len(note) - 1
            if body_budget > 0:
                body = output[:body_budget].rstrip()
                result = f"{header_text}\n{body}{note}"
            else:
                result = f"{header_text}{note}"

    return result


class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取本地【纯文本】文件内容。"
            "支持全文读取、行范围、关键词过滤、上下文与字符截断。"
        )

    @property
    def parameters(self) -> dict:
        return FILE_READER_TOOLS[0]["function"]["parameters"]

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return True

    @property
    def allow_spill(self) -> bool:
        return False

    async def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        pattern: Optional[str] = None,
        context_lines: int = 0,
        encoding: str = "utf-8",
        max_chars: Optional[int] = None,
    ) -> str:
        return read_file(
            path=path,
            start_line=start_line,
            end_line=end_line,
            pattern=pattern,
            context_lines=context_lines,
            encoding=encoding,
            max_chars=max_chars,
        )
