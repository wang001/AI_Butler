"""
命令执行器（容器内直接执行）

AI Butler 整体运行在 Docker 容器中，命令直接通过 subprocess 执行。
容器本身就是隔离层，无需"沙箱中的沙箱"。

安全策略由容器层面保证：
  - 容器以非 root 用户运行
  - 宿主机通过只读/读写挂载共享数据目录
  - 容器 --cap-drop ALL，无特权
  - 资源限制由 docker-compose.yml 或 docker run 参数控制

与外部机器的数据交互通过共享目录实现：
  - /data          → 持久化数据（记忆、对话日志等）
  - /workspace     → 工作区（用户文件、脚本执行等）
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandConfig:
    """命令执行配置"""
    # 工作目录（容器内路径）
    workdir: str = "/workspace"
    # 超时（秒）
    default_timeout: int = 30
    max_timeout: int = 120


class CommandExecutor:
    """容器内命令执行器"""

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
          - success: bool
          - stdout: str
          - stderr: str
          - exit_code: int
          - timed_out: bool
          - error: str | None
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
