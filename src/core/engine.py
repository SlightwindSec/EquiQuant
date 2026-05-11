import os
import json
import shutil
from datetime import datetime

from .strategy import TuningStrategy
from ..modules.aqt import AutomaticQuantizationTool
from ..modules.quantizer import ModelslimQuantizer, LLMCompressorQuantizer
from ..modules.server import VllmServer
from ..modules.bencher import AisBencher
from ..utils.logger import logger


QUANTIZER_MAPPING = {
    "msmodelslim": ModelslimQuantizer,
    "llmcompressor": LLMCompressorQuantizer,
}

QUANT_CONFIG_NAME = {
    "msmodelslim": "generated_modelslim_config.yaml",
    "llmcompressor": "generated_llmcompressor_config.py",
}


class EquiQuantEngine:
    def __init__(self, config: dict):
        self.config = config
        self.workspace = config["workspace"]
        self.evaluation_config = config["evaluation"]
        self.strategy = TuningStrategy(
            initial_fallback=config["strategy"]["initial_fallback_layers"]
        )
        default_tolerance = self.evaluation_config.get("tolerance_ratio", 1.00)
        self.target_accuracies = {}
        for name, data in self.evaluation_config.get("datasets", {}).items():
            tolerance = data.get("tolerance_ratio", default_tolerance)
            self.target_accuracies[name] = {
                "target_accuracy": data["target_accuracy"],
                "tolerance_ratio": tolerance,
            }
        self.last_results = None
        self.run_id = 0
        self.quantized_model_path = ""

        self.quantization_tool = config.get("quantization_tool", "msmodelslim")

        # AQT 相关
        self.aqt_config = config.get("aqt") or {}
        self.aqt_tool = None
        self.current_budget = None
        self._init_aqt()

    def run(self):
        logger.info("Starting EquiQuant optimization loop...")
        self._run_with_aqt()

    def _persist_successful_run(self, quantized_model_path, quant_config_path):
        best_base_dir = os.path.join(
            self.workspace["base_dir"], self.workspace["best_weights_dir"]
        )
        os.makedirs(best_base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(
            best_base_dir, f"trial_{self.run_id:03d}_{timestamp}"
        )
        os.makedirs(archive_dir, exist_ok=True)

        try:
            if os.path.isdir(quantized_model_path):
                dest = os.path.join(archive_dir, os.path.basename(quantized_model_path))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(quantized_model_path, dest)
            elif os.path.isfile(quantized_model_path):
                shutil.copy2(
                    quantized_model_path,
                    os.path.join(archive_dir, os.path.basename(quantized_model_path)),
                )
            else:
                logger.warning(
                    f"Quantized weights path does not exist: {quantized_model_path}"
                )
        except Exception as exc:
            logger.error(f"Failed to archive quantized weights: {exc}")

        if os.path.exists(quant_config_path):
            try:
                shutil.copy2(
                    quant_config_path,
                    os.path.join(archive_dir, os.path.basename(quant_config_path)),
                )
            except Exception as exc:
                logger.warning(f"Failed to copy quant config: {exc}")

        metadata = {
            "run_id": self.run_id,
            "fallback_layers": list(self.strategy.current_fallback),
            "results": self.last_results,
            "targets": self.target_accuracies,
            "timestamp": datetime.now().isoformat(),
        }

        metadata_path = os.path.join(archive_dir, "metrics.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
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
            base_model_path=self.config["base_model_path"],
            workspace=self.workspace,
        )

    def _assess_results(self, results):
        """
        返回每个数据集的状态：low / ok / high / missing
        """
        status = {}
        for name, target_cfg in self.target_accuracies.items():
            value = results.get(name) if results else None
            if value is None:
                status[name] = "missing"
                continue
            lower, upper = self._calculate_bounds(target_cfg)
            if value < lower:
                status[name] = "low"
            elif value > upper:
                status[name] = "high"
            else:
                status[name] = "ok"
            logger.info(f"Dataset {name}: value={value:.4f}, bounds=[{lower:.4f}, {upper:.4f}], status={status[name]}")
        return status

    def _calculate_bounds(self, target_cfg):
        if isinstance(target_cfg, dict):
            target = target_cfg.get("target_accuracy") or target_cfg.get("target")
            tolerance = target_cfg.get("tolerance_ratio") or target_cfg.get(
                "tolerance", 0.01
            )
        else:
            target = target_cfg
            tolerance = 0.01

        if target is None:
            raise ValueError("Target accuracy must be provided for each dataset.")

        # Relative tolerance: 0.01 means 1% of target
        delta = target * tolerance
        lower = target - delta
        upper = target + delta
        return lower, upper

    # ------------------------------------------------------------------ #
    # Workflows
    # ------------------------------------------------------------------ #
    def _run_with_aqt(self):
        # Step 0: Sensitivity Analysis
        logger.info(f"\n{'=' * 20} Sensitivity Analysis (AQT) {'=' * 20}")
        if os.path.exists(self.aqt_tool.sensitivity_scores_save_path):
            logger.info("Sensitivity scores already exist, skipping analysis.")
        else:
            logger.info("Computing sensitivity scores.")
            self.aqt_tool.compute_sensitivity_scores_only()
            if not os.path.exists(self.aqt_tool.sensitivity_scores_save_path):
                logger.error("Sensitivity scores missing. Analysis failed.")
                return

        # Adaptive Search Tuning Parameters
        base_layer_step = int(self.aqt_config.get("base_layer_step", 5))
        min_step = int(self.aqt_config.get("min_layer_step", 1))
        max_step = int(self.aqt_config.get("max_layer_step", 15))
        gap_threshold = float(self.aqt_config.get("gap_threshold", 0.05)) # Gap > 5% is "large"
        
        # Reset stateful config
        if os.path.exists(self.aqt_tool.layer_configs_path):
            os.remove(self.aqt_tool.layer_configs_path)

        while True:
            self.run_id += 1
            logger.info(f"\n{'=' * 20} Trial {self.run_id} (Adaptive Search) {'=' * 20}")
            current_run_dir = os.path.join(
                self.workspace["base_dir"], self.workspace["current_run_dir"]
            )
            os.makedirs(current_run_dir, exist_ok=True)
            quant_config_path = os.path.join(
                current_run_dir, QUANT_CONFIG_NAME[self.quantization_tool]
            )
            quant_weights_path = os.path.join(
                current_run_dir, self.workspace["quant_weights_dir"]
            )

            try:
                # Step 1: AQT Logic (Stateful)
                hybrid_quant_schema_path, hybrid_quant_schema_re_path = (
                    self.aqt_tool.run(run_id=self.run_id)
                )
                if not os.path.exists(hybrid_quant_schema_path):
                    logger.error("AQT failed to generate schema. Stopping.")
                    break

                # Step 2: Quantization
                quant_log_path = os.path.join(current_run_dir, f"{self.quantization_tool}.log")
                quantizer_cls = QUANTIZER_MAPPING[self.quantization_tool]
                quantizer = quantizer_cls(
                    quant_config=self.config["quantization"],
                    base_model_path=self.config["base_model_path"],
                    fallback_layers=self.config.get("disable_names"),
                    output_config_path=quant_config_path,
                    output_weights_path=quant_weights_path,
                    hybrid_quant_schema_path=hybrid_quant_schema_path,
                    hybrid_quant_schema_re_path=hybrid_quant_schema_re_path,
                    quant_log_path=quant_log_path,
                )

                quantized_model_path = quantizer.run()
                if not quantized_model_path or not os.path.exists(quantized_model_path):
                    logger.error("Quantization failed. Stopping.")
                    break

                # Step 3: Serving
                vllm_log_path = os.path.join(current_run_dir, "vllm_server.log")
                server = VllmServer(
                    model_path=quantized_model_path,
                    server_config=self.config["vllm_server"],
                    log_file_path=vllm_log_path,
                )

                if not server.start():
                    logger.error("VLLM failed to start. Stopping.")
                    break

                # Step 4: Benchmarking with Early Stopping
                def check_accuracy(alias, acc):
                    target_cfg = self.target_accuracies.get(alias)
                    lower, _ = self._calculate_bounds(target_cfg)
                    return acc >= lower

                bencher = AisBencher(
                    eval_config=self.evaluation_config,
                    server_config=self.config["vllm_server"],
                    quantized_model_path=quantized_model_path,
                    current_run_dir=current_run_dir,
                    run_id=self.run_id,
                )
                self.last_results = bencher.run(early_stop_fn=check_accuracy)
                logger.info(f"Trial {self.run_id} Results (Partial/Full): {self.last_results}")

                # Step 5: Adaptive Assessment
                status = self._assess_results(self.last_results)
                # is_passed is True if all datasets meet or exceed the accuracy floor (ok/high)
                is_passed = all(v in ("ok", "high") for v in status.values())

                if is_passed:
                    self._persist_successful_run(quantized_model_path, quant_config_path)
                    logger.info("Target accuracy satisfied (all datasets ok or high). Optimization complete.")
                    break

                # Calculate Gap and Adaptive Step Size K
                # Note: Gap is only calculated for evaluated datasets; missing ones are ignored.
                max_gap = 0.0
                evaluated_gaps = []
                for name, target_cfg in self.target_accuracies.items():
                    if name in self.last_results:
                        target = target_cfg.get("target_accuracy", 0.0)
                        current = self.last_results[name]
                        gap = target - current
                        evaluated_gaps.append(gap)
                
                if evaluated_gaps:
                    max_gap = max(evaluated_gaps)
                else:
                    # Fallback if somehow no results were returned
                    max_gap = gap_threshold 
                
                # Adaptive K logic
                k = int(base_layer_step * (max_gap / gap_threshold))
                k = max(min_step, min(k, max_step))
                
                logger.info(f"Accuracy shortfall: {max_gap:.4f}. Adaptive step size K = {k} layers.")
                
                # Apply Upgrades to State
                self.aqt_tool.apply_upgrades(k)
                # Note: Next iteration will use the updated layer_configs_path in self.aqt_tool.run()

            finally:
                if "server" in locals() and server.process.process:
                    server.stop()
