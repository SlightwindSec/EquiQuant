import os
import json
import shutil
from datetime import datetime

from core.strategy import TuningStrategy
from modules.quantizer import ModelslimQuantizer
from modules.server import VllmServer
from modules.bencher import AisBencher
from utils.logger import logger
from utils.model_inspector import ModelInspector


class EquiQuantEngine:
    def __init__(self, config):
        self.config = config
        self.workspace = config['workspace']
        self.evaluation_config = config['evaluation']
        self.all_layers = ModelInspector.get_layers_by_pattern(
            config['base_model_path'], 
            config['strategy']['layer_pattern']
        )
        self.strategy = TuningStrategy(
            all_possible_layers=self.all_layers,
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

    def run(self):
        logger.info("Starting EquiQuant optimization loop...")

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
