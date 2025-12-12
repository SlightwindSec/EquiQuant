import os
import json
import shutil
from datetime import datetime

from core.strategy import TuningStrategy
from modules.aqt import AutomaticQuantizationTool
from modules.quantizer import ModelslimQuantizer
from modules.server import VllmServer
from modules.bencher import AisBencher
from utils.logger import logger


class EquiQuantEngine:
    def __init__(self, config):
        self.config = config
        self.workspace = config['workspace']
        self.evaluation_config = config['evaluation']
        self.strategy = TuningStrategy(
            initial_fallback=config['strategy']['initial_fallback_layers']
        )
        default_tolerance = self.evaluation_config.get('tolerance_ratio', 0.01)
        self.target_accuracies = {}
        for name, data in self.evaluation_config.get('datasets', {}).items():
            tolerance = data.get('tolerance_ratio', default_tolerance)
            self.target_accuracies[name] = {
                'target_accuracy': data['target_accuracy'],
                'tolerance_ratio': tolerance
            }
        self.last_results = None
        self.run_id = 0

        # AQT 相关
        self.aqt_config = self.config.get('aqt') or {}
        self.aqt_tool = None
        self.use_aqt = False
        self.current_budget = None
        self._init_aqt()

    def run(self):
        logger.info("Starting EquiQuant optimization loop...")
        if self.use_aqt:
            self._run_with_aqt()
        else:
            self._run_without_aqt()

    def _persist_successful_run(self, quantized_model_path, quant_config_path):
        best_base_dir = os.path.join(self.workspace['base_dir'], self.workspace['best_weights_dir'])
        os.makedirs(best_base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(best_base_dir, f"trial_{self.run_id:03d}_{timestamp}")
        os.makedirs(archive_dir, exist_ok=True)

        try:
            if os.path.isdir(quantized_model_path):
                dest = os.path.join(archive_dir, os.path.basename(quantized_model_path))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(quantized_model_path, dest)
            elif os.path.isfile(quantized_model_path):
                shutil.copy2(quantized_model_path, os.path.join(archive_dir, os.path.basename(quantized_model_path)))
            else:
                logger.warning(f"Quantized weights path does not exist: {quantized_model_path}")
        except Exception as exc:
            logger.error(f"Failed to archive quantized weights: {exc}")

        if os.path.exists(quant_config_path):
            try:
                shutil.copy2(quant_config_path, os.path.join(archive_dir, os.path.basename(quant_config_path)))
            except Exception as exc:
                logger.warning(f"Failed to copy quant config: {exc}")

        metadata = {
            "run_id": self.run_id,
            "fallback_layers": list(self.strategy.current_fallback),
            "results": self.last_results,
            "targets": self.target_accuracies,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(archive_dir, "metrics.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Failed to write metadata file {metadata_path}: {exc}")

        logger.info(f"Saved successful artifacts to {archive_dir}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _init_aqt(self):
        enabled_in_cfg = self.aqt_config.get('enabled', True)
        if not enabled_in_cfg:
            logger.info("AQT disabled by config.")
            return

        self.aqt_tool = AutomaticQuantizationTool(
            aqt_config=self.aqt_config,
            base_model_path=self.config['base_model_path'],
            workspace=self.workspace
        )
        self.use_aqt = self.aqt_tool.is_available()
        if self.use_aqt:
            self.current_budget = self.aqt_tool.initial_budget
            logger.info(f"AQT detected. Initial checkpoint budget: {self.current_budget} MB")
        else:
            logger.info("AQT not available, fallback to built-in strategy.")

    def _assess_results(self, results):
        """
        返回每个数据集的状态：low / ok / high / missing
        """
        status = {}
        for name, target_cfg in self.target_accuracies.items():
            value = results.get(name) if results else None
            if value is None:
                status[name] = 'missing'
                continue
            lower, upper = self._calculate_bounds(target_cfg)
            if value < lower:
                status[name] = 'low'
            elif value > upper:
                status[name] = 'high'
            else:
                status[name] = 'ok'
        return status

    def _calculate_bounds(self, target_cfg):
        if isinstance(target_cfg, dict):
            target = target_cfg.get('target_accuracy') or target_cfg.get('target')
            tolerance = target_cfg.get('tolerance_ratio') or target_cfg.get('tolerance', 0.01)
        else:
            target = target_cfg
            tolerance = 0.0

        if target is None:
            raise ValueError("Target accuracy must be provided for each dataset.")

        lower = target * (1 - tolerance)
        upper = target * (1 + tolerance)
        return lower, upper

    # ------------------------------------------------------------------ #
    # Workflows
    # ------------------------------------------------------------------ #
    def _run_without_aqt(self):
        template = self.config['quantization']['template_config']
        v1_config = template.get('v1')
        api_version = template.get('apiversion')
        
        if v1_config or api_version == 'modelslim_v1':
            logger.warning(
                "modelslim_v1 format is detected in non-AQT mode. "
                "v1 format is primarily designed for AQT scenarios. "
                "For non-AQT scenarios, please use v0 format (legacy, but currently supported)."
            )
        
        while True:
            self.run_id += 1
            logger.info(f"\n{'='*20} Trial {self.run_id} {'='*20}")
            current_run_dir = os.path.join(self.workspace['base_dir'], self.workspace['current_run_dir'])
            os.makedirs(current_run_dir, exist_ok=True)
            fallback_layers = self.strategy.next_trial(self.last_results, self.target_accuracies)

            if isinstance(fallback_layers, str):
                logger.info(f"Stopping loop: {fallback_layers}")
                break

            logger.info(f"Running trial {self.run_id} with {len(fallback_layers)} fallback layers.")
            quant_config_path = os.path.join(current_run_dir, self.workspace['quant_config_name'])
            quant_weights_path = os.path.join(current_run_dir, self.workspace['quant_weights_dir'])

            try:
                quantizer = ModelslimQuantizer(
                    quant_config=self.config['quantization'],
                    base_model_path=self.config['base_model_path'],
                    fallback_layers=fallback_layers,
                    output_config_path=quant_config_path,
                    output_weights_path=quant_weights_path
                )
                quantized_model_path = quantizer.run()
                if not quantized_model_path:
                    logger.error("Quantization failed. Skipping this trial.")
                    break

                vllm_log_path = os.path.join(current_run_dir, "vllm_server.log")

                server = VllmServer(
                    model_path=quantized_model_path,
                    server_config=self.config['vllm_server'],
                    log_file_path=vllm_log_path
                )

                if not server.start():
                    logger.error("VLLM failed to start, skipping this trial.")
                    continue # server.stop() 已经在 start() 失败时被调用了

                bencher = AisBencher(
                    eval_config=self.evaluation_config,
                    server_config=self.config['vllm_server'],
                    quantized_model_path=quantized_model_path,
                    current_run_dir=current_run_dir,
                    run_id=self.run_id
                )
                self.last_results = bencher.run()
                logger.info(f"Trial {self.run_id} Results: {self.last_results}")

                passed, failed = self.strategy.evaluate_results(self.last_results, self.target_accuracies)
                if passed:
                    self.strategy.best_result = {'fallback': list(fallback_layers), 'results': self.last_results}
                    self._persist_successful_run(quantized_model_path, quant_config_path)
                    logger.info("Target accuracy satisfied within tolerance. Exiting optimization loop.")
                    break
                else:
                    logger.info(f"Datasets outside tolerance: {failed}")

            finally:
                if 'server' in locals() and server.process.process:
                    server.stop()

    def _run_with_aqt(self):
        if not self.use_aqt:
            logger.error("AQT workflow invoked without availability.")
            return

        # 在 AQT 模式下，回退层完全由 AQT 的敏感度结果决定，这里不主动指定。
        base_disable_names = []

        while True:
            self.run_id += 1
            logger.info(f"\n{'='*20} Trial {self.run_id} (AQT) {'='*20}")
            current_run_dir = os.path.join(self.workspace['base_dir'], self.workspace['current_run_dir'])
            os.makedirs(current_run_dir, exist_ok=True)

            quant_config_path = os.path.join(current_run_dir, self.workspace['quant_config_name'])
            base_config_path = os.path.join(current_run_dir, f"base_{self.workspace['quant_config_name']}")
            quant_weights_path = os.path.join(current_run_dir, self.workspace['quant_weights_dir'])

            try:
                # step1: 生成基础 modelslim YAML
                base_quantizer = ModelslimQuantizer(
                    quant_config=self.config['quantization'],
                    base_model_path=self.config['base_model_path'],
                    fallback_layers=base_disable_names,
                    output_config_path=base_config_path,
                    output_weights_path=quant_weights_path
                )
                base_cfg = base_quantizer.generate_config_only(disable_names=base_disable_names, output_path=base_config_path)
                if not base_cfg:
                    logger.error("Failed to generate base modelslim config for AQT.")
                    break

                # step2: 调用 AQT 生成最终 YAML
                final_config = self.aqt_tool.generate_modelslim_config(
                    run_id=self.run_id,
                    base_config_path=base_cfg,
                    budget_mb=self.current_budget,
                    output_config_path=quant_config_path
                )
                if not final_config:
                    logger.error("AQT failed to produce final modelslim config.")
                    break

                # step3: 量化
                quantizer = ModelslimQuantizer(
                    quant_config=self.config['quantization'],
                    base_model_path=self.config['base_model_path'],
                    fallback_layers=base_disable_names,
                    output_config_path=quant_config_path,
                    output_weights_path=quant_weights_path,
                    prepared_config_path=final_config
                )
                quantized_model_path = quantizer.run()
                if not quantized_model_path:
                    logger.error("Quantization failed. Skipping this trial.")
                    break

                vllm_log_path = os.path.join(current_run_dir, "vllm_server.log")
                server = VllmServer(
                    model_path=quantized_model_path,
                    server_config=self.config['vllm_server'],
                    log_file_path=vllm_log_path
                )

                if not server.start():
                    logger.error("VLLM failed to start, skipping this trial.")
                    continue

                bencher = AisBencher(
                    eval_config=self.evaluation_config,
                    server_config=self.config['vllm_server'],
                    quantized_model_path=quantized_model_path,
                    current_run_dir=current_run_dir,
                    run_id=self.run_id
                )
                self.last_results = bencher.run()
                logger.info(f"Trial {self.run_id} Results: {self.last_results}")

                status = self._assess_results(self.last_results)
                logger.info(f"AQT assessment: {status}")
                if any(v in ('missing', 'low') for v in status.values()):
                    new_budget = self.aqt_tool.increase_budget(self.current_budget)
                    if new_budget == self.current_budget:
                        logger.error("Reached max AQT budget but accuracy is still low. Stopping.")
                        break
                    logger.info(f"Accuracy below target. Increasing AQT budget to {new_budget} MB.")
                    self.current_budget = new_budget
                    continue

                if all(v == 'ok' for v in status.values()):
                    self.strategy.best_result = {'fallback': list(base_disable_names), 'results': self.last_results}
                    self._persist_successful_run(quantized_model_path, quant_config_path)
                    logger.info("Target accuracy satisfied within tolerance. Exiting optimization loop.")
                    break

                # 剩余情况：全部高于上界，尝试降低预算以提升性能
                if all(v == 'high' for v in status.values()):
                    new_budget = self.aqt_tool.decrease_budget(self.current_budget)
                    if new_budget == self.current_budget:
                        logger.info("Budget already at minimum, accept current results.")
                        self._persist_successful_run(quantized_model_path, quant_config_path)
                        break
                    logger.info(f"Accuracy well above target, decreasing AQT budget to {new_budget} MB for better performance.")
                    self.current_budget = new_budget
                    continue

                # 混合高/ok 情况：认为满足要求
                self.strategy.best_result = {'fallback': list(base_disable_names), 'results': self.last_results}
                self._persist_successful_run(quantized_model_path, quant_config_path)
                logger.info("Mixed OK/HIGH results accepted. Exiting optimization loop.")
                break

            finally:
                if 'server' in locals() and server.process.process:
                    server.stop()
