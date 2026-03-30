import subprocess
import os
import signal
from .logger import logger


class ShellRunner:
    @staticmethod
    def run_cmd(cmd, timeout=None, log_path=None):
        logger.info(f"Executing: {cmd}")

        log_content = f"===== Command executed at: {os.popen('date').read().strip()} =====\n"
        log_content += f"Command: {cmd}\n"
        log_content += "=" * 80 + "\n"
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
                encoding="utf-8",
                errors="replace",
            )

            log_content += f"Return code: {result.returncode}\n\n"
            log_content += "----- STDOUT -----\n"
            log_content += result.stdout + "\n\n"
            log_content += "----- STDERR -----\n"
            log_content += result.stderr + "\n"

            if log_path:
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(log_content)
                    logger.info(f"Command log saved to: {log_path}")
                except Exception as e:
                    logger.error(f"Failed to write log to {log_path}: {str(e)}")

            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
            return result.returncode == 0, result.stdout, result.stderr
        
        except subprocess.TimeoutExpired:
            log_content += "ERROR: Command timed out!\n"
            if log_path:
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(log_content)
                except Exception as e:
                    logger.error(f"Failed to write timeout log to {log_path}: {str(e)}")
            
            logger.error("Command timed out")
            return False, "", "Timeout"


class AsyncProcess:
    """用于管理 vllm 这种需要长期运行的服务进程"""

    def __init__(self, cmd, log_file):
        self.cmd = cmd
        self.process = None
        self.log_file = open(log_file, "w")

    def start(self):
        logger.info(f"Starting Async Process: {self.cmd}")
        # 使用 preexec_fn=os.setsid 创建新的进程组，方便后续能够杀掉整个进程树
        self.process = subprocess.Popen(
            self.cmd,
            shell=True,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            executable="/bin/bash",
        )

    def stop(self):
        if self.process:
            logger.info(f"Stopping process PID: {self.process.pid}")
            try:
                # 发送 SIGTERM 给整个进程组，确保子进程也能被杀掉
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception as e:
                logger.warning(f"Normal stop failed, forcing kill: {e}")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass  # 进程可能已经不在了
        self.log_file.close()
