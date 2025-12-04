import copy
import os
import re
import time
import importlib.util
from pathlib import Path
from typing import Dict, Optional

from utils.shell import ShellRunner
from utils.logger import logger


class AisBencher:
    """
    负责生成 ais_bench 所需的 model 配置文件，并对每个数据集执行评测。
    """

    def __init__(self, eval_config, server_config, quantized_model_path, current_run_dir, run_id):
        """
        Args:
            eval_config (dict): config.yaml['evaluation']
            server_config (dict): config.yaml['vllm_server']
            quantized_model_path (str): 本轮量化权重的保存目录/文件
            current_run_dir (str): 工作区当前运行目录，用于保存日志
            run_id (int): 当前 Trial 编号
        """
        self.eval_config = eval_config or {}
        self.server_config = server_config or {}
        self.quantized_model_path = quantized_model_path
        self.current_run_dir = current_run_dir
        self.run_id = run_id

        self.ais_config = self.eval_config.get('aisbench', {})
        self.datasets = self.eval_config.get('datasets', {})
        self.disable_thinking = bool(self.eval_config.get('disable_qwen_thinking', False))

        self.binary = self.ais_config.get('binary', 'ais_bench')
        self.global_mode = self.ais_config.get('mode', 'all')
        self.timeout = self.ais_config.get('timeout', 7200) # 2 小时默认
        self.cleanup_generated_config = self.ais_config.get('cleanup_model_config', True)

        self.model_cfg_meta = self.ais_config.get('model_config', {})
        self.model_config_dir: Optional[Path] = None
        self.model_config_name: Optional[str] = None
        self.model_config_path: Optional[Path] = None

        self.default_request_rate = self.ais_config.get('request_rate', 1)
        self.default_metric_keys = self.ais_config.get('default_metric_keys', ['final_accuracy', 'accuracy', 'score'])

        # vLLM service 信息
        self.served_model_name = self._resolve_served_model_name()
        self.host_ip = self.ais_config.get('host_ip') or self.server_config.get('host', 'localhost')
        self.host_port = self.ais_config.get('host_port') or self.server_config.get('port')

        if not self.datasets:
            logger.warning("config['evaluation']['datasets'] is empty. Nothing to benchmark.")

    def run(self) -> Dict[str, float]:
        """
        运行所有配置的数据集评测。
        Returns:
            dict: {dataset_alias: accuracy}
        """
        results = {}
        if not self.datasets:
            return results

        try:
            self._prepare_model_config_handle()
        except Exception as exc:
            logger.error(f"Failed to prepare ais_bench model config directory: {exc}")
            return results

        logger.info(f"Starting AISBench evaluation for {len(self.datasets)} dataset(s)...")

        try:
            for alias, dataset_cfg in self.datasets.items():
                logger.info(f"[AISBench] Dataset: {alias}")
                request_rate = dataset_cfg.get('request_rate', self.default_request_rate)
                try:
                    self._write_model_config(request_rate=request_rate)
                except Exception as exc:
                    logger.error(f"Could not generate ais_bench model config for {alias}: {exc}")
                    results[alias] = 0.0
                    continue

                try:
                    cmd = self._build_command(dataset_cfg)
                except Exception as exc:
                    logger.error(f"Invalid dataset config for {alias}: {exc}")
                    results[alias] = 0.0
                    continue

                success, stdout, stderr = ShellRunner.run_cmd(cmd, timeout=self.timeout)
                log_file = self._write_log(alias, cmd, stdout, stderr)

                if not success:
                    logger.error(f"AISBench command failed for {alias}. See log: {log_file}")
                    results[alias] = 0.0
                    continue

                accuracy = self._parse_accuracy(stdout + "\n" + stderr, alias, dataset_cfg)
                logger.info(f"AISBench result for {alias}: {accuracy}")
                results[alias] = accuracy
        finally:
            self._cleanup_model_config()

        return results

    def _prepare_model_config_handle(self):
        """
        定位 ais_bench 的 models 配置目录，并生成当前 run 专属的配置名。
        """
        explicit_dir = self.model_cfg_meta.get('directory')
        if explicit_dir:
            self.model_config_dir = Path(explicit_dir)
        else:
            spec = importlib.util.find_spec("ais_bench")
            if not spec or not spec.submodule_search_locations:
                raise ImportError("Cannot locate ais_bench package on PYTHONPATH.")
            base_dir = Path(spec.submodule_search_locations[0])
            subdir = self.model_cfg_meta.get('subdir', 'vllm_api')
            self.model_config_dir = base_dir / "benchmark" / "configs" / "models" / subdir

        if not self.model_config_dir.exists():
            raise FileNotFoundError(f"Model config directory not found: {self.model_config_dir}")

        base_name = self.model_cfg_meta.get('base_name', 'vllm_api_general_chat')
        base_name = base_name.replace('.py', '')
        suffix = self.model_cfg_meta.get('name_suffix', 'auto')
        if suffix in (None, '', 'auto'):
            suffix = f"trial{self.run_id:03d}_{int(time.time())}"
        suffix = re.sub(r'[^0-9A-Za-z_]+', '_', suffix)

        self.model_config_name = f"{base_name}_{suffix}"
        self.model_config_path = self.model_config_dir / f"{self.model_config_name}.py"
        logger.info(f"AISBench model config will be written to: {self.model_config_path}")

    def _write_model_config(self, request_rate: Optional[float] = None):
        """
        根据当前量化结果生成 vllm_api model config。
        """
        if not self.model_config_path:
            raise RuntimeError("Model config path has not been prepared.")

        postproc = self.ais_config.get('pred_postprocessor', 'extract_non_reasoning_content')
        if postproc:
            import_line = f"from ais_bench.benchmark.utils.model_postprocessors import {postproc}\n"
            postproc_field = f"\n        pred_postprocessor=dict(type={postproc})"
        else:
            import_line = ""
            postproc_field = ""

        retry = self.ais_config.get('retry', 2)
        batch_size = self.ais_config.get('batch_size', 1)
        max_out_len = self.ais_config.get('max_out_len', 512)
        trust_remote_code = self.ais_config.get('trust_remote_code', False)
        abbr = self.model_cfg_meta.get('abbr', 'vllm-api-general-chat')
        attr = self.model_cfg_meta.get('attr', 'service')
        generation_defaults = {
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "seed": None,
            "repetition_penalty": 1.03,
        }
        generation_kwargs = copy.deepcopy(self.ais_config.get('generation_kwargs', generation_defaults))
        if self.disable_thinking:
            chat_kwargs = generation_kwargs.setdefault("chat_template_kwargs", {})
            chat_kwargs["enable_thinking"] = False

        request_rate_value = request_rate if request_rate is not None else self.default_request_rate
        host_port_value = self.host_port
        try:
            host_port_value = int(host_port_value)
        except (TypeError, ValueError):
            pass

        content = (
            "from ais_bench.benchmark.models import VLLMCustomAPIChat\n"
            f"{import_line}"
            "models = [\n"
            "    dict(\n"
            f"        attr={repr(attr)},\n"
            "        type=VLLMCustomAPIChat,\n"
            f"        abbr={repr(abbr)},\n"
            f"        path={repr(self.quantized_model_path)},\n"
            f"        model={repr(self.served_model_name)},\n"
            f"        request_rate={request_rate_value},\n"
            f"        retry={retry},\n"
            f"        host_ip={repr(self.host_ip)},\n"
            f"        host_port={repr(host_port_value)},\n"
            f"        max_out_len={max_out_len},\n"
            f"        batch_size={batch_size},\n"
            f"        trust_remote_code={trust_remote_code},\n"
            f"        generation_kwargs={repr(generation_kwargs)},{postproc_field}\n"
            "    )\n"
            "]\n"
        )

        with open(self.model_config_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _cleanup_model_config(self):
        if self.cleanup_generated_config and self.model_config_path and self.model_config_path.exists():
            try:
                self.model_config_path.unlink()
                logger.debug(f"Removed temporary AISBench model config: {self.model_config_path}")
            except Exception as exc:
                logger.warning(f"Failed to remove AISBench model config {self.model_config_path}: {exc}")

    def _build_command(self, dataset_cfg: Dict) -> str:
        dataset_cli_name = dataset_cfg.get('config_name')
        if not dataset_cli_name:
            raise KeyError("Dataset config must provide 'config_name' for ais_bench.")

        mode = dataset_cfg.get('mode', self.global_mode)

        extra_args = []
        global_extra = self.ais_config.get('extra_args')
        dataset_extra = dataset_cfg.get('extra_args')
        for candidate in (global_extra, dataset_extra):
            if not candidate:
                continue
            if isinstance(candidate, str):
                extra_args.append(candidate.strip())
            elif isinstance(candidate, (list, tuple)):
                extra_args.extend(str(item).strip() for item in candidate if item)

        extra_str = " ".join(arg for arg in extra_args if arg)

        cmd = f"{self.binary} --models {self.model_config_name} --datasets {dataset_cli_name} --mode {mode}"
        if extra_str:
            cmd = f"{cmd} {extra_str}"
        return cmd

    def _write_log(self, dataset_alias: str, cmd: str, stdout: str, stderr: str) -> str:
        log_dir = self.ais_config.get('log_dir') or os.path.join(self.current_run_dir, "aisbench_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{dataset_alias}_trial{self.run_id:03d}.log")

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"[COMMAND]\n{cmd}\n\n[STDOUT]\n{stdout}\n\n[STDERR]\n{stderr}\n")

        logger.info(f"AISBench logs for {dataset_alias} saved to: {log_path}")
        return log_path

    def _parse_accuracy(self, logs: str, dataset_alias: str, dataset_cfg: Dict) -> float:
        patterns = []

        custom_regex = dataset_cfg.get('result_regex')
        if custom_regex:
            patterns.append(custom_regex)

        metric_keys = dataset_cfg.get('metric_keys') or []
        for key in metric_keys:
            patterns.append(rf"{re.escape(key)}\s*[:=]\s*([\d\.]+)%?")

        for key in self.default_metric_keys:
            patterns.append(rf"{re.escape(key)}\s*[:=]\s*([\d\.]+)%?")

        patterns.append(r"accuracy\s*[:=]\s*([\d\.]+)%?")
        patterns.append(r"score\s*[:=]\s*([\d\.]+)%?")

        for pattern in patterns:
            matches = re.findall(pattern, logs, flags=re.IGNORECASE)
            if matches:
                value = matches[-1]
                try:
                    return float(value.strip("% "))
                except ValueError:
                    continue

        logger.warning(f"Could not parse accuracy from AISBench logs for {dataset_alias}. Defaulting to 0.0.")
        return 0.0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _resolve_served_model_name(self) -> str:
        args = self.server_config.get('args', {})
        for key in ('served-model-name', 'served_model_name'):
            if key in args and args[key]:
                return args[key]
        raise KeyError("vllm_server.args must provide 'served-model-name' for AISBench.")
