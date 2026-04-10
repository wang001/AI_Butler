"""
cli/stream.py — 终端流式渲染

ThinkingSpinner : "Butler 思考中…" 旋转指示器，支持暂停（打印工具进度时用）
StreamPrinter   : 逐 token 打印到终端，safe_print 兼容 prompt_toolkit patch_stdout
"""
import sys
from contextlib import contextmanager


def safe_print(*args, sep=" ", end="\n"):
    """线程安全输出，兼容 prompt_toolkit patch_stdout 上下文。"""
    sys.stdout.write(sep.join(str(a) for a in args) + end)
    sys.stdout.flush()


class ThinkingSpinner:
    """
    "Butler 思考中…" 旋转指示器。

    用法：
        with ThinkingSpinner():
            reply = await butler.chat(text)
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Butler 思考中"):
        self._msg = message
        self._task = None

    def __enter__(self):
        import asyncio, threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=1)
        # 清除旋转行
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self):
        import time
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r{frame} {self._msg}…")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    @contextmanager
    def pause(self):
        """暂停旋转以打印工具进度，打印完毕后自动恢复。"""
        self._stop.set()
        self._thread.join(timeout=1)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        try:
            yield
        finally:
            self._stop = __import__("threading").Event()
            self._thread = __import__("threading").Thread(target=self._spin, daemon=True)
            self._thread.start()
