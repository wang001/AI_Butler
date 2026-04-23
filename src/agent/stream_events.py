# -*- coding: utf-8 -*-
"""
兼容层：流式事件定义已迁移到 event/stream.py。

保留本文件仅用于兼容旧导入路径，后续新代码应直接从 event 包导入。
"""

from event.stream import StreamEvent

__all__ = ["StreamEvent"]
