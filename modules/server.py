import time
import requests
import json
import shlex
from utils.shell import AsyncProcess
from utils.logger import logger


class VllmServer:
    """
    配置驱动的 VLLM-Ascend 服务器启动器。
    1. 从配置中构建环境变量 (env_vars)。
    2. 从配置中构建命令行参数 (args)，处理 bool, str, dict。
    3. 启动服务并等待其就绪 (health check)。
    4. 停止服务。
    """

    def __init__(self, model_path, server_config, log_file_path):
        """
        Args:
            model_path (str): (动态) 量化后模型的路径。
            server_config (dict): 来自 config.yaml['vllm_server'] 的配置。
            log_file_path (str): (动态) 本次运行的 vllm 日志文件路径。
        """
        self.config = server_config
        self.model_path = model_path
        self.log_file = log_file_path

        # 构造健康检查 URL
        self.health_check_url = (
            f"http://{self.config['host']}:{self.config['port']}"
            f"{self.config['health_check_endpoint']}"
        )
        self.startup_timeout = self.config["startup_timeout"]

        # 构造完整的 shell 命令
        self.cmd = self._build_command()

        # 初始化异步进程管理器
        self.process = AsyncProcess(self.cmd, self.log_file)
        logger.debug(f"VLLM command constructed: {self.cmd}")

    def _build_env_prefix(self):
        """
        从 env_vars 配置构建 'export K1=V1; export K2=V2; ' 字符串。
        """
        env_vars = self.config.get("env_vars", {})
        if not env_vars:
            return ""

        parts = []
        for key, value in env_vars.items():
            # shlex.quote 确保值被正确转义，例如 "0.11.0"
            parts.append(f"export {key}={shlex.quote(str(value))};")
        return " ".join(parts) + " "  # 结尾的空格

    def _build_command(self):
        """
        构建完整的 python -m vllm... 命令，包括所有参数。
        """
        # 1. 环境变量
        env_prefix = self._build_env_prefix()

        # 2. Python 入口点
        entrypoint = self.config["entrypoint"]
        cmd_parts = [f"python -m {entrypoint}"]

        # 3. 添加动态/核心参数
        cmd_parts.append(f"--model {shlex.quote(self.model_path)}")
        cmd_parts.append(f"--port {self.config['port']}")

        # 4. 遍历配置中的 'args' 来构建其他参数
        for key, value in self.config.get("args", {}).items():
            if value is True:
                # e.g., trust-remote-code: true -> --trust-remote-code
                cmd_parts.append(f"--{key}")
            elif value is False or value is None:
                # e.g., enable-prefix-caching: false -> (被忽略)
                continue
            elif isinstance(value, dict):
                # e.g., additional_config: {...} -> --additional_config='{"...":...}'
                # 序列化为紧凑的 JSON, 并使用 shlex.quote 添加外层引号
                json_str = json.dumps(value, separators=(",", ":"))
                cmd_parts.append(f"--{key}={shlex.quote(json_str)}")
            else:
                # e.g., tp: 2 -> --tp 2
                cmd_parts.append(f"--{key} {shlex.quote(str(value))}")

        full_command_str = " ".join(cmd_parts)
        return env_prefix + full_command_str

    def start(self):
        """启动 VLLM 进程并等待其就绪。"""
        logger.info(f"Starting VLLM server for model: {self.model_path}")
        self.process.start()

        logger.info(f"Waiting for server to be ready at {self.health_check_url} ...")
        if not self._wait_for_ready():
            logger.error(f"VLLM server failed to start. Check log: {self.log_file}")
            # 尝试停止僵尸进程
            try:
                self.stop()
            except:
                pass
            return False

        logger.info("VLLM server started successfully.")
        return True

    def stop(self):
        """停止 VLLM 进程。"""
        logger.info("Stopping VLLM server...")
        self.process.stop()
        logger.info("VLLM server stopped.")

    def _wait_for_ready(self):
        """轮询健康检查接口。"""
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                resp = requests.get(self.health_check_url, timeout=3)
                if resp.status_code == 200:
                    return True
            except requests.ConnectionError:
                # 服务器尚未监听端口
                pass
            except Exception as e:
                # 其他错误 (e.g., 500 internal server error during init)
                logger.debug(f"Health check received non-connection error: {e}")

            time.sleep(5)  # 5秒轮询间隔

        logger.error(f"Server startup timed out after {self.startup_timeout} seconds.")
        return False
