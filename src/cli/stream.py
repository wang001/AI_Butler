"""
cli/stream.py — 终端流式渲染

ThinkingSpinner : "Butler 思考中…" 旋转指示器，支持暂停（打印工具进度时用）
StreamPrinter   : 逐 token 打印到终端，safe_print 兼容 prompt_toolkit patch_stdout
"""
import shutil
import sys
import threading
from contextlib import contextmanager


_STDOUT_LOCK = threading.Lock()


def _clear_current_line(width: int = 0) -> None:
    """
    清除当前行，兼容 Docker attach / TTY 环境。
    
    策略：
    1. 先输出 \r 回到行首
    2. 输出足够多的空格覆盖整行
    3. 再输出 \r 回到行首（为下一行输出做准备）
    """
    columns = shutil.get_terminal_size(fallback=(80, 20)).columns
    blank_width = max(width, columns, 80)  # 至少 80 个空格，确保覆盖
    sys.stdout.write("\r" + (" " * blank_width) + "\r")
    sys.stdout.flush()


def safe_print(*args, sep=" ", end="\n"):
    """线程安全输出，兼容 prompt_toolkit patch_stdout 上下文。"""
    with _STDOUT_LOCK:
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
        self._last_render_width = 0

    def __enter__(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.stop(clear=True)

    def stop(self, clear: bool = True) -> None:
        """
        停止 spinner。
        
        关键：
        1. 设置 _stop 事件，通知 spinner 线程停止
        2. 等待线程完全退出（join）
        3. 清除当前行（如果 clear=True）
        
        这确保了 spinner 线程完全停止后才进行清除，避免竞态条件。
        """
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if clear:
            with _STDOUT_LOCK:
                _clear_current_line(self._last_render_width)
                sys.stdout.flush()

    def _spin(self):
        """
        Spinner 线程主循环。
        
        关键：在每次输出前检查 _stop 事件，确保停止后不再输出。
        """
        import time
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            text = f"{frame} {self._msg}…"
            self._last_render_width = len(text)
            with _STDOUT_LOCK:
                # 再次检查 _stop，避免在获得锁后才停止导致的输出
                if not self._stop.is_set():
                    _clear_current_line(self._last_render_width)
                    sys.stdout.write(text)
                    sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    @contextmanager
    def pause(self):
        """
        暂停旋转以打印工具进度，打印完毕后自动恢复。
        
        关键：
        1. 停止当前 spinner 线程
        2. 执行用户代码（yield）
        3. 创建新的 spinner 线程并启动
        """
        self.stop(clear=True)
        try:
            yield
        finally:
            # 重新初始化 spinner 线程
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
