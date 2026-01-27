import copy
import os
import re
import time
import json
import csv
import shutil
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Tuple

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
                    cmd = self._build_command(alias, dataset_cfg)
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

                # 尝试从输出文件中解析结果
                accuracy, result_data = self._parse_results_from_files(stdout + "\n" + stderr, alias, dataset_cfg)
                if accuracy is None:
                    # 如果文件解析失败，回退到日志解析
                    accuracy = self._parse_accuracy(stdout + "\n" + stderr, alias, dataset_cfg)
                    logger.warning(f"Failed to parse results from files for {alias}, using log parsing fallback.")

                logger.info(f"AISBench result for {alias}: {accuracy}")
                results[alias] = accuracy

                # 保存解析的结果数据
                if result_data:
                    self._save_result_data(alias, result_data)
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

        # 根据是否使用 Chat 模版，选择不同的 AISBench Model 类型
        use_chat_template = bool(self.model_cfg_meta.get('use_chat_template', True))
        model_type = self.model_cfg_meta.get('type')
        if not model_type:
            model_type = 'VLLMCustomAPIChat' if use_chat_template else 'VLLMCustomAPI'

        if model_type == 'VLLMCustomAPIChat':
            model_import = "from ais_bench.benchmark.models import VLLMCustomAPIChat\n"
        elif model_type == 'VLLMCustomAPI':
            model_import = "from ais_bench.benchmark.models import VLLMCustomAPI\n"
        else:
            # 兜底：未知类型时仍然回退到 Chat 版本，避免直接崩溃
            model_import = "from ais_bench.benchmark.models import VLLMCustomAPIChat\n"
            model_type = 'VLLMCustomAPIChat'

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
            f"{model_import}"
            f"{import_line}"
            "models = [\n"
            "    dict(\n"
            f"        attr={repr(attr)},\n"
            f"        type={model_type},\n"
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

    def _build_command(self, dataset_alias: str, dataset_cfg: Dict) -> str:
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

        # 特殊数据集需要合并多个子数据集时，追加 --merge-ds
        # 这里同时对数据集别名和 config_name 做判断，以兼容不同配置风格。
        alias_lower = (dataset_alias or "").lower()
        cli_name_lower = str(dataset_cli_name).lower()
        if any(key in alias_lower for key in ("ceval", "mmlu")) or any(
            key in cli_name_lower for key in ("ceval", "mmlu")
        ):
            cmd = f"{cmd} --merge-ds"

        return cmd

    def _write_log(self, dataset_alias: str, cmd: str, stdout: str, stderr: str) -> str:
        log_dir = self.ais_config.get('log_dir') or os.path.join(self.current_run_dir, "aisbench_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{dataset_alias}_trial{self.run_id:03d}.log")

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"[COMMAND]\n{cmd}\n\n[STDOUT]\n{stdout}\n\n[STDERR]\n{stderr}\n")

        logger.info(f"AISBench logs for {dataset_alias} saved to: {log_path}")
        return log_path

    def _extract_output_path(self, logs: str) -> Optional[str]:
        """
        从日志中提取 AISBench 的输出路径。
        查找类似 "Current exp folder: outputs/default/20250628_151326" 的日志。
        
        Args:
            logs: 标准输出和标准错误的合并内容
            
        Returns:
            输出路径（相对路径或绝对路径），如果未找到则返回 None
        """
        # 匹配 "Current exp folder: outputs/default/20250628_151326" 这样的模式
        pattern = r"Current exp folder:\s*([^\s\n]+)"
        match = re.search(pattern, logs, re.IGNORECASE)
        if match:
            output_path = match.group(1).strip()
            # 如果是相对路径，需要基于当前工作目录解析
            if not os.path.isabs(output_path):
                # 尝试从命令执行的工作目录查找
                cwd = os.getcwd()
                abs_path = os.path.join(cwd, output_path)
                if os.path.exists(abs_path):
                    return abs_path
                return output_path
            return output_path
        return None

    def _parse_results_from_files(self, logs: str, dataset_alias: str, dataset_cfg: Dict) -> Tuple[Optional[float], Optional[Dict]]:
        """
        从 AISBench 的输出文件中解析结果。
        
        Args:
            logs: 标准输出和标准错误的合并内容
            dataset_alias: 数据集别名
            dataset_cfg: 数据集配置
            
        Returns:
            (accuracy, result_data) 元组，如果解析失败则返回 (None, None)
            result_data 包含 summary, results, predictions 等信息
        """
        output_path = self._extract_output_path(logs)
        if not output_path:
            logger.debug(f"Could not extract output path from logs for {dataset_alias}")
            return None, None

        if not os.path.exists(output_path):
            logger.warning(f"AISBench output path does not exist: {output_path}")
            return None, None

        result_data = {
            'output_path': output_path,
            'summary': None,
            'results': None,
            'predictions': None
        }

        # 解析 summary CSV 文件
        summary_data = self._parse_summary_csv(output_path, dataset_alias, dataset_cfg)
        if summary_data:
            result_data['summary'] = summary_data
            accuracy = summary_data.get('accuracy')
            if accuracy is not None:
                # 同时解析 results JSON 和 predictions JSON
                result_data['results'] = self._parse_results_json(output_path, dataset_alias, dataset_cfg)
                result_data['predictions'] = self._parse_predictions_json(output_path, dataset_alias, dataset_cfg)
                return accuracy, result_data

        return None, None

    def _parse_summary_csv(self, output_path: str, dataset_alias: str, dataset_cfg: Dict) -> Optional[Dict]:
        """
        解析 summary CSV 文件。
        
        Args:
            output_path: AISBench 输出目录路径
            dataset_alias: 数据集别名
            dataset_cfg: 数据集配置
            
        Returns:
            包含解析结果的字典，如果解析失败则返回 None
        """
        summary_dir = os.path.join(output_path, 'summary')
        if not os.path.exists(summary_dir):
            logger.warning(f"Summary directory not found: {summary_dir}")
            return None

        # 查找 summary CSV 文件（格式：summary_YYYYMMDD_HHMMSS.csv）
        csv_files = list(Path(summary_dir).glob('summary_*.csv'))
        if not csv_files:
            logger.warning(f"No summary CSV file found in {summary_dir}")
            return None

        # 使用最新的 CSV 文件
        csv_file = sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    logger.warning(f"Summary CSV file is empty: {csv_file}")
                    return None

                # 查找匹配的数据集行
                # CSV 格式: dataset,version,metric,mode,vllm-api-general
                # 示例: cevaldataset,-,accuracy,gen,78.97
                dataset_name = dataset_cfg.get('config_name', dataset_alias)
                metric_keys = dataset_cfg.get('metric_keys', [])

                # 尝试匹配数据集名称（可能不完全匹配，需要模糊匹配）
                matched_row = None
                for row in rows:
                    row_dataset = row.get('dataset', '').lower()
                    # 检查数据集名称是否匹配（支持部分匹配）
                    if (dataset_name.lower() in row_dataset or 
                        row_dataset in dataset_name.lower() or
                        dataset_alias.lower() in row_dataset):
                        matched_row = row
                        break

                # 如果没找到精确匹配，使用第一行（通常只有一个数据集）
                if not matched_row and rows:
                    matched_row = rows[0]

                if not matched_row:
                    logger.warning(f"No matching dataset found in summary CSV for {dataset_alias}")
                    return None

                # 提取精度值
                # CSV 的最后一列通常是精度值（列名可能是模型缩写，如 vllm-api-general）
                accuracy = None
                for key, value in matched_row.items():
                    if key.lower() in ('dataset', 'version', 'metric', 'mode'):
                        continue
                    try:
                        accuracy = float(value)
                        break
                    except (ValueError, TypeError):
                        continue

                if accuracy is None:
                    logger.warning(f"Could not extract accuracy from summary CSV row: {matched_row}")
                    return None

                return {
                    'csv_file': str(csv_file),
                    'dataset': matched_row.get('dataset', ''),
                    'metric': matched_row.get('metric', ''),
                    'mode': matched_row.get('mode', ''),
                    'accuracy': accuracy,
                    'raw_row': matched_row
                }

        except Exception as exc:
            logger.error(f"Failed to parse summary CSV file {csv_file}: {exc}")
            return None

    def _parse_results_json(self, output_path: str, dataset_alias: str, dataset_cfg: Dict) -> Optional[Dict]:
        """
        解析 results JSON 文件。
        
        Args:
            output_path: AISBench 输出目录路径
            dataset_alias: 数据集别名
            dataset_cfg: 数据集配置
            
        Returns:
            包含解析结果的字典，如果解析失败则返回 None
        """
        results_dir = os.path.join(output_path, 'results')
        if not os.path.exists(results_dir):
            logger.debug(f"Results directory not found: {results_dir}")
            return None

        # 查找 results JSON 文件
        # 路径格式: results/vllm-api-general-chat/dataset_name.json
        # 需要遍历子目录查找
        json_files = []
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        if not json_files:
            logger.debug(f"No results JSON file found in {results_dir}")
            return None

        # 尝试匹配数据集名称
        dataset_name = dataset_cfg.get('config_name', dataset_alias)
        matched_file = None

        for json_file in json_files:
            file_name = os.path.basename(json_file).replace('.json', '').lower()
            if (dataset_name.lower() in file_name or 
                file_name in dataset_name.lower() or
                dataset_alias.lower() in file_name):
                matched_file = json_file
                break

        # 如果没找到匹配，使用第一个 JSON 文件
        if not matched_file and json_files:
            matched_file = json_files[0]

        if not matched_file:
            return None

        try:
            with open(matched_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'json_file': matched_file,
                    'data': data
                }
        except Exception as exc:
            logger.warning(f"Failed to parse results JSON file {matched_file}: {exc}")
            return None

    def _parse_predictions_json(self, output_path: str, dataset_alias: str, dataset_cfg: Dict) -> Optional[Dict]:
        """
        解析 predictions JSON 文件（用于将来分析哪些题目做错）。
        
        Args:
            output_path: AISBench 输出目录路径
            dataset_alias: 数据集别名
            dataset_cfg: 数据集配置
            
        Returns:
            包含解析结果的字典，如果解析失败则返回 None
        """
        predictions_dir = os.path.join(output_path, 'predictions')
        if not os.path.exists(predictions_dir):
            logger.debug(f"Predictions directory not found: {predictions_dir}")
            return None

        # 查找 predictions JSON 文件
        json_files = []
        for root, dirs, files in os.walk(predictions_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        if not json_files:
            logger.debug(f"No predictions JSON file found in {predictions_dir}")
            return None

        # 尝试匹配数据集名称
        dataset_name = dataset_cfg.get('config_name', dataset_alias)
        matched_file = None

        for json_file in json_files:
            file_name = os.path.basename(json_file).replace('.json', '').lower()
            if (dataset_name.lower() in file_name or 
                file_name in dataset_name.lower() or
                dataset_alias.lower() in file_name):
                matched_file = json_file
                break

        # 如果没找到匹配，使用第一个 JSON 文件
        if not matched_file and json_files:
            matched_file = json_files[0]

        if not matched_file:
            return None

        try:
            with open(matched_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'json_file': matched_file,
                    'data': data
                }
        except Exception as exc:
            logger.warning(f"Failed to parse predictions JSON file {matched_file}: {exc}")
            return None

    def _save_result_data(self, dataset_alias: str, result_data: Dict):
        """
        保存解析的结果数据到当前运行目录。
        
        Args:
            dataset_alias: 数据集别名
            result_data: 包含 summary, results, predictions 等信息的字典
        """
        save_dir = os.path.join(self.current_run_dir, "aisbench_results", f"trial{self.run_id:03d}")
        os.makedirs(save_dir, exist_ok=True)

        dataset_save_dir = os.path.join(save_dir, dataset_alias)
        os.makedirs(dataset_save_dir, exist_ok=True)

        # 保存 summary 信息
        if result_data.get('summary'):
            summary_file = os.path.join(dataset_save_dir, 'summary.json')
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data['summary'], f, indent=2, ensure_ascii=False)
                logger.debug(f"Saved summary data to {summary_file}")
            except Exception as exc:
                logger.warning(f"Failed to save summary data: {exc}")

        # 保存 results JSON（原始分数）
        if result_data.get('results'):
            results_file = os.path.join(dataset_save_dir, 'results.json')
            try:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data['results'], f, indent=2, ensure_ascii=False)
                logger.debug(f"Saved results data to {results_file}")
            except Exception as exc:
                logger.warning(f"Failed to save results data: {exc}")

        # 保存 predictions JSON（推理结果，将来用于分析错题）
        if result_data.get('predictions'):
            predictions_file = os.path.join(dataset_save_dir, 'predictions.json')
            try:
                # predictions 可能很大，直接复制原文件而不是重新序列化
                source_file = result_data['predictions'].get('json_file')
                if source_file and os.path.exists(source_file):
                    shutil.copy2(source_file, predictions_file)
                    logger.debug(f"Saved predictions data to {predictions_file}")
                else:
                    # 如果源文件不存在，则序列化数据
                    with open(predictions_file, 'w', encoding='utf-8') as f:
                        json.dump(result_data['predictions'], f, indent=2, ensure_ascii=False)
                    logger.debug(f"Saved predictions data to {predictions_file}")
            except Exception as exc:
                logger.warning(f"Failed to save predictions data: {exc}")
        
        # 保存元数据
        metadata = {
            'dataset_alias': dataset_alias,
            'output_path': result_data.get('output_path'),
            'run_id': self.run_id,
            'has_summary': result_data.get('summary') is not None,
            'has_results': result_data.get('results') is not None,
            'has_predictions': result_data.get('predictions') is not None
        }
        metadata_file = os.path.join(dataset_save_dir, 'metadata.json')
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Failed to save metadata: {exc}")

    def _parse_accuracy(self, logs: str, dataset_alias: str, dataset_cfg: Dict) -> float:
        """
        从日志中解析精度值（作为文件解析失败时的回退方案）。
        """
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
