import os
import json
import shutil
from datetime import datetime

from core.strategy import TuningStrategy
from modules.aqt import AutomaticQuantizationTool
from modules.quantizer import ModelslimQuantizer, LLMCompressorQuantizer
from modules.server import VllmServer
from modules.bencher import AisBencher
from utils.logger import logger


QUANTIZER_MAPPING = {
    "msmodelslim": ModelslimQuantizer,
    "llmcompressor": LLMCompressorQuantizer
}

QUANT_CONFIG_NAME = {
    "msmodelslim": "generated_modelslim_config.yaml",
    "llmcompressor": "generated_llmcompressor_config.py"
}


class EquiQuantEngine:
    def __init__(self, config):
        self.config = config
        self.workspace = config['workspace']
        self.evaluation_config = config['evaluation']
        self.strategy = TuningStrategy(
            initial_fallback=config['strategy']['initial_fallback_layers']
        )
        default_tolerance = self.evaluation_config.get('tolerance_ratio', 1.00)
        self.target_accuracies = {}
        for name, data in self.evaluation_config.get('datasets', {}).items():
            tolerance = data.get('tolerance_ratio', default_tolerance)
            self.target_accuracies[name] = {
                'target_accuracy': data['target_accuracy'],
                'tolerance_ratio': tolerance
            }
        self.last_results = None
        self.last_hybrid_quant_schema_path = ""
        self.run_id = 0

        self.quantization_tool = config.get('quantization_tool', 'msmodelslim')

        # AQT 相关
        self.aqt_config = config.get('aqt') or {}
        self.aqt_tool = None
        self.current_budget = None
        self._init_aqt()

    def run(self):
        logger.info("Starting EquiQuant optimization loop...")
        self._run_with_aqt()

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
        self.aqt_tool = AutomaticQuantizationTool(
            aqt_config=self.aqt_config,
            base_model_path=self.config['base_model_path'],
            workspace=self.workspace,
        )
        self.current_budget = self.aqt_tool.initial_budget
        logger.info(f"AQT detected. Initial checkpoint budget: {self.current_budget} MB")

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
            logger.info(f"[!!!] {lower=}, {upper=}, {status=}")
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
            tolerance = target_cfg.get('tolerance_ratio') or target_cfg.get('tolerance', 1.00)
        else:
            target = target_cfg
            tolerance = 0.0

        if target is None:
            raise ValueError("Target accuracy must be provided for each dataset.")

        lower = target - tolerance
        upper = target + tolerance
        return lower, upper

    # ------------------------------------------------------------------ #
    # Workflows
    # ------------------------------------------------------------------ #
    def _run_with_aqt(self):
        base_disable_names = self.config.get('disable_names')

        # Step 0: 调用 AQT 进行敏感度分析
        logger.info(f"\n{'='*20} Sensitivity Alalysis (AQT) {'='*20}")
        self.aqt_tool.compute_sensitivity_scores_only()
        if not os.path.exists(self.aqt_tool.sensitivity_scores_save_path):
            logger.error(f"Sensitivity_scores_save_path: {self.aqt_tool.sensitivity_scores_save_path} does not exist.")
            return

        while True:
            self.run_id += 1
            logger.info(f"\n{'='*20} Trial {self.run_id} (AQT) {'='*20}")
            current_run_dir = os.path.join(self.workspace['base_dir'], self.workspace['current_run_dir'])
            os.makedirs(current_run_dir, exist_ok=True)
            quant_config_path = os.path.join(current_run_dir, QUANT_CONFIG_NAME[self.quantization_tool])
            quant_weights_path = os.path.join(current_run_dir, self.workspace['quant_weights_dir'])

            try:
                # Step 1: 调用 AQT 获取量化配置路径
                hybrid_quant_schema_path, hybrid_quant_schema_re_path = self.aqt_tool.run(
                    run_id=self.run_id,
                    budget_mb=self.current_budget,
                    last_hybrid_quant_schema_path=self.last_hybrid_quant_schema_path,
                )
                if not os.path.exists(hybrid_quant_schema_path) or not os.path.exists(hybrid_quant_schema_re_path):
                    logger.error("AQT failed. Skipping this trial.")
                    break
                self.last_hybrid_quant_schema_path = hybrid_quant_schema_path

                # Step 2: 量化器获取量化配置生成量化所需yaml/py
                quantizer_cls = QUANTIZER_MAPPING[self.quantization_tool]
                quantizer = quantizer_cls(
                    quant_config=self.config['quantization'],
                    base_model_path=self.config['base_model_path'],
                    fallback_layers=base_disable_names,
                    output_config_path=quant_config_path,
                    output_weights_path=quant_weights_path,
                    hybrid_quant_schema_path=hybrid_quant_schema_path,
                    hybrid_quant_schema_re_path=hybrid_quant_schema_re_path,
                )

                # Step 3: 量化器量化模型
                quantized_model_path = quantizer.run()
                if not quantized_model_path:
                    logger.error("Quantization failed. Skipping this trial.")
                    break

                # Step 4: 拉取vllm服务
                vllm_log_path = os.path.join(current_run_dir, "vllm_server.log")
                server = VllmServer(
                    model_path=quantized_model_path,
                    server_config=self.config['vllm_server'],
                    log_file_path=vllm_log_path
                )

                if not server.start():
                    logger.error("VLLM failed to start, skipping this trial.")
                    continue

                # Step 5: aisbench评测
                bencher = AisBencher(
                    eval_config=self.evaluation_config,
                    server_config=self.config['vllm_server'],
                    quantized_model_path=quantized_model_path,
                    current_run_dir=current_run_dir,
                    run_id=self.run_id
                )
                self.last_results = bencher.run()
                logger.info(f"Trial {self.run_id} Results: {self.last_results}")

                # Step 6: 结果评估，预算调整
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
