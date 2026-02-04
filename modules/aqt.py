import os
import shlex
from typing import Dict, Optional, Tuple

from utils.logger import logger
from utils.shell import ShellRunner


class AutomaticQuantizationTool:
    """
    封装 AQT 的两步调用：
    1) 计算敏感度，生成 mse_scores.json
    2) 基于模板 + 敏感度，输出最终的 modelslim YAML
    """

    def __init__(self, aqt_config: dict, base_model_path: str, workspace: dict, quant_type: str="minmax"):
        self.config = aqt_config or {}
        self.base_model_path = os.path.abspath(base_model_path)
        self.workspace = workspace or {}
        self.quant_type = quant_type

        self.metric = self.config.get('sensitivity_metric', 'mse')
        self.results_root = os.path.abspath(
            self.config.get('results_dir') or os.path.join(self.workspace.get('base_dir', 'workspace'), "aqt_results")
        )
        self.initial_budget = int(self.config.get('initial_budget_mb', 2500))
        self.max_budget = int(self.config.get('max_budget_mb', self.initial_budget))
        self.min_budget = int(self.config.get('min_budget_mb', 0))
        self.step_up = int(self.config.get('budget_step_mb', 500))
        self.step_down = int(self.config.get('budget_step_down_mb', self.step_up))
        self.omp_num_threads = int(self.config.get('omp_num_threads', 32))
        self.visible_devices = str(self.config.get('ascend_visible_devices', "0"))

    def increase_budget(self, current: int) -> int:
        return min(current + self.step_up, self.max_budget)

    def decrease_budget(self, current: int) -> int:
        return max(current - self.step_down, self.min_budget)

    # ------------------------------------------------------------------ #
    # AQT 执行
    # ------------------------------------------------------------------ #
    def _build_save_dir(self, run_id: int, budget_mb: int) -> str:
        save_dir = os.path.join(self.results_root, f"run{run_id:03d}", f"budget_{budget_mb}mb")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _hybrid_quant_schema_path(self, save_dir: str) -> str:
        return os.path.join(save_dir, "hybrid_quant_schema.json")

    def _analysis_cmd(
        self,
        save_dir: str,
        budget_mb: int,
        quant_data_path: str,
        quant_data_save_path: str,
        last_hybrid_quant_schema_path: str,
    ) -> str:
        cmd = (
            f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(self.visible_devices)}; "
            f"export OMP_NUM_THREADS={self.omp_num_threads}; "
            f"python aqt/launch.py "
            f"--model-name-or-path {shlex.quote(self.base_model_path)} "
            f"--quant-data-path {shlex.quote(quant_data_path)} "
            f"--quant-data-save-path {shlex.quote(quant_data_save_path)} "
            f"--quant-samples-num {self.config.get('quant_samples_num', 128)} "
            f"--quant-context-length {self.config.get('quant_context_length', 4096)} "
            f"--quant-type {self.quant_type} "
            f"--quant-sym "
            f"--disable-smoothquant "
            f"--sensitivity-metric {shlex.quote(self.metric)} "
            f"--compute-sensitivity-scores-only "
            f"--ckpt-size-budget-mb {budget_mb} "
            f"--save-dir {shlex.quote(save_dir)} "
            f"--results_root {self.results_root} "
            f"--last_hybrid_quant_schema_path {shlex.quote(last_hybrid_quant_schema_path)} "
            f"--eval-ppl "
        )
        return cmd


    def run(
        self,
        run_id: int,
        budget_mb: int,
        last_hybrid_quant_schema_path: str = ""
    ) -> str:
        """
        运行 AQT 获取敏感度分析得到的量化配置。
        """
        quant_data_path = self.config.get('quant_data_path')
        if not quant_data_path:
            logger.error("AQT requires `aqt_quant_data_path` in config.")
            return None
        quant_data_path = os.path.abspath(quant_data_path)

        save_dir = self._build_save_dir(run_id, budget_mb)
        quant_data_save_path = os.path.join(self.results_root, "calib_data.pt")
        os.makedirs(os.path.dirname(quant_data_save_path), exist_ok=True)

        # 敏感度分析
        analysis_cmd = self._analysis_cmd(save_dir, budget_mb, quant_data_path, quant_data_save_path, last_hybrid_quant_schema_path)
        success, stdout, stderr = ShellRunner.run_cmd(analysis_cmd, timeout=10800)
        if not success:
            logger.error("AQT sensitivity analysis failed.")
            return None

        hybrid_quant_schema_path = self._hybrid_quant_schema_path(save_dir)
        if not os.path.exists(hybrid_quant_schema_path):
            logger.error(f"AQT output missing after prepare step: {hybrid_quant_schema_path}")
            return None

        return hybrid_quant_schema_path
