import subprocess
import os
import signal
from .logger import logger


class ShellRunner:
    @staticmethod
    def run_cmd(cmd, timeout=None, log_path=None):
        logger.info(f"Executing: {cmd}")
        log_file = None
        if log_path:
            try:
                log_file = open(log_path, "w", encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to open log file {log_path}: {e}")

        stdout_lines = []
        process = None
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                text=True,
                executable="/bin/bash",
                encoding="utf-8",
                errors="replace",
                start_new_session=True
            )

            for line in process.stdout:
                stdout_lines.append(line)
                if log_file:
                    log_file.write(line)
                    log_file.flush()

            process.wait(timeout=timeout)
            return_code = process.returncode

            if return_code != 0:
                logger.error(f"Command failed with code {return_code}")

            return return_code == 0, "".join(stdout_lines), ""

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds: {cmd}")

            if process:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            
            return False, "".join(stdout_lines), "Timeout"

        except Exception as e:
            logger.error(f"Command execution error: {str(e)}")
            return False, "".join(stdout_lines), f"Exception: {str(e)}"

        finally:
            if log_file:
                log_file.close()
            if process and process.stdout:
                process.stdout.close()


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
