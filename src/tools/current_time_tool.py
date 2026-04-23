"""
tools/current_time_tool.py — 当前时间工具
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from tools.base import Tool


CURRENT_TIME_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "返回当前的日期和时间。用户询问时间、日期、星期时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "tz_offset": {
                    "type": "number",
                    "description": (
                        "UTC 偏移小时数，默认 8（北京时间）。"
                        "例如：东京 9、伦敦 0、纽约 -5、洛杉矶 -8。"
                        "支持半小时时区，如印度 5.5、尼泊尔 5.75。"
                    ),
                    "default": 8,
                },
            },
            "required": [],
        },
    },
}


def get_current_time(tz_offset: float = 8) -> str:
    """返回指定时区的当前时间。tz_offset 为 UTC 偏移小时数，默认 8（北京时间）。"""
    tz = timezone(timedelta(hours=tz_offset))
    now = datetime.now(tz)
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekdays[now.weekday()]
    offset_mins = int(tz_offset * 60)
    sign = "+" if offset_mins >= 0 else "-"
    h, m = divmod(abs(offset_mins), 60)
    tz_label = f"UTC{sign}{h}" if m == 0 else f"UTC{sign}{h}:{m:02d}"
    return now.strftime(f"%Y年%m月%d日 {wd} %H:%M:%S（{tz_label}）")


class CurrentTimeTool(Tool):
    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "返回当前的日期和时间。用户询问时间、日期、星期时调用。"

    @property
    def parameters(self) -> dict:
        return CURRENT_TIME_TOOL["function"]["parameters"]

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return True

    async def execute(self, tz_offset: float = 8) -> str:
        return get_current_time(tz_offset=tz_offset)
