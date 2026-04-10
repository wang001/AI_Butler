"""
命令执行器（Docker 容器内 subprocess 直接执行）

AI Butler 运行在 Docker 容器中，容器本身即隔离层，
命令通过 asyncio.create_subprocess_shell 直接执行，无需额外沙箱。

安全边界由 Docker 层面保证：
  - 容器以非 root 用户运行
  - --cap-drop ALL，无特权
  - 宿主机目录通过只读/读写 volume 挂载，不暴露其他路径
  - 资源限制（CPU / 内存）由 docker-compose.yml 或 docker run 参数控制

容器内共享目录约定：
  - /data       → 持久化数据（记忆、对话日志、向量库）
  - /workspace  → 工作区（Agent 命令执行、用户文件读写）
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path

# ── OpenAI function-calling schema ────────────────────────────────────────────

COMMAND_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "在容器环境中执行 shell 命令。"
                "用于文件操作、数据处理、代码运行、系统信息查询等。"
                "工作目录为 /workspace，可读写。"
                "容器本身提供安全隔离，命令不会影响宿主机。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令，如 'ls -la' 或 'python3 script.py'",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时秒数，默认 30，最大 120",
                        "default": 30,
                    },
                    "workdir": {
                        "type": "string",
                        "description": "执行命令的工作目录，默认 /workspace",
                        "default": "/workspace",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


@dataclass
class CommandConfig:
    """命令执行配置"""
    # 默认工作目录（容器内路径）
    workdir: str = "/workspace"
    # 默认超时（秒）
    default_timeout: int = 30
    # 允许的最大超时上限
    max_timeout: int = 120


class CommandExecutor:
    """Docker 容器内 shell 命令执行器。"""

    def __init__(self, config: CommandConfig | None = None):
        self.config = config or CommandConfig()
        # 确保工作目录存在
        Path(self.config.workdir).mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> dict:
        """
        在容器内执行 shell 命令。

        返回 dict：
          - success   : bool
          - stdout    : str
          - stderr    : str
          - exit_code : int
          - timed_out : bool
          - error     : str | None
        """
        effective_timeout = min(
            timeout or self.config.default_timeout,
            self.config.max_timeout,
        )
        effective_workdir = workdir or self.config.workdir

        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_workdir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=effective_timeout,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace").strip(),
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
                "exit_code": proc.returncode,
                "timed_out": False,
                "error": None,
            }
        except asyncio.TimeoutError:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            return {
                "success": False, "stdout": "", "stderr": "",
                "exit_code": -1, "timed_out": True,
                "error": f"命令执行超时（{effective_timeout}秒限制）",
            }
        except Exception as e:
            return {
                "success": False, "stdout": "", "stderr": "",
                "exit_code": -1, "timed_out": False,
                "error": f"执行异常: {e}",
            }
