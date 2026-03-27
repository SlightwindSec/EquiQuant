import os
import shlex

from utils.logger import logger
from utils.shell import ShellRunner


class AutomaticQuantizationTool:
    """
    封装 AQT 的两步调用：
    1) 计算敏感度，生成 mse_scores.json
    2) 基于模板 + 敏感度，输出最终的 modelslim YAML
    """

    def __init__(
        self,
        aqt_config: dict,
        base_model_path: str,
        workspace: dict,
        quant_type: str = "minmax",
    ):
        self.config = aqt_config or {}
        self.base_model_path = os.path.abspath(base_model_path)
        self.workspace = workspace or {}
        self.quant_type = quant_type

        self.metrics = self.config.get("sensitivity_metrics", ["mse"])
        self.results_root = os.path.abspath(
            self.config.get("results_dir")
            or os.path.join(self.workspace.get("base_dir", "workspace"), "aqt_results")
        )
        self.initial_budget = int(self.config.get("initial_budget_mb", 2500))
        self.max_budget = int(self.config.get("max_budget_mb", self.initial_budget))
        self.min_budget = int(self.config.get("min_budget_mb", 0))
        self.step_up = int(self.config.get("budget_step_mb", 500))
        self.step_down = int(self.config.get("budget_step_down_mb", self.step_up))
        self.omp_num_threads = int(self.config.get("omp_num_threads", 32))
        self.visible_devices = str(self.config.get("ascend_visible_devices", "0"))

    def increase_budget(self, current: int) -> int:
        return min(current + self.step_up, self.max_budget)

    def decrease_budget(self, current: int) -> int:
        return max(current - self.step_down, self.min_budget)

    # ------------------------------------------------------------------ #
    # AQT 执行
    # ------------------------------------------------------------------ #
    def _compute_sensitivity_scores_cmd(
        self,
        save_dir: str,
        quant_data_path: str,
        quant_data_save_path: str,
    ) -> str:
        cmd = (
            f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(self.visible_devices)}; "
            f"export OMP_NUM_THREADS={self.omp_num_threads}; "
            f"python aqt/compute_sensitivity_scores.py "
            f"--model-name-or-path {shlex.quote(self.base_model_path)} "
            f"--seed 42 "
            f"--quant-data-path {shlex.quote(quant_data_path)} "
            f"--quant-data-save-path {shlex.quote(quant_data_save_path)} "
            f"--quant-samples-num {self.config.get('quant_samples_num', 128)} "
            f"--quant-context-length {self.config.get('quant_context_length', 4096)} "
            f"--quant-type {self.quant_type} "
            f"--sensitivity-metrics {shlex.quote(','.join(self.metrics))} "
            f"--save-dir {shlex.quote(save_dir)} "
            f"--sensitivity_scores_save_path {shlex.quote(self.sensitivity_scores_save_path)} "
            f"--is-mm {self.config.get('is_mm', False)} "
            f"--is-deepseek-v32 {self.config.get('is_deepseek_v32', False)} "
        )
        return cmd

    def compute_sensitivity_scores_only(self, flag: bool = True):
        save_dir = os.path.join(self.results_root, self.quant_type)
        os.makedirs(save_dir, exist_ok=True)
        self.sensitivity_scores_save_path = os.path.join(
            save_dir, "sensitivity_scores.json"
        )

        quant_data_path = self.config.get("quant_data_path")
        if not quant_data_path:
            logger.error("AQT requires `aqt_quant_data_path` in config.")
            return None
        quant_data_path = os.path.abspath(quant_data_path)
        quant_data_save_path = os.path.join(self.results_root, "calib_data.pt")
        os.makedirs(os.path.dirname(quant_data_save_path), exist_ok=True)

        if flag:
            compute_sensitivity_scores_cmd = self._compute_sensitivity_scores_cmd(
                save_dir, quant_data_path, quant_data_save_path
            )
            success, stdout, stderr = ShellRunner.run_cmd(
                compute_sensitivity_scores_cmd, timeout=10800
            )
            if not success or not os.path.exists(self.sensitivity_scores_save_path):
                logger.error("Computing sensitivity scores failed.")
        else:
            logger.info("Sensitivity analysis skipped!")

    def _run_cmd(
        self,
        budget_mb: int,
        hybrid_quant_schema_path: str,
        hybrid_quant_schema_re_path: str,
    ) -> str:
        cmd = (
            f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(self.visible_devices)}; "
            f"export OMP_NUM_THREADS={self.omp_num_threads}; "
            f"python aqt/launch.py "
            f"--model-name-or-path {shlex.quote(self.base_model_path)} "
            f"--ckpt-size-budget-mb {budget_mb} "
            f"--hybrid_quant_schema_path {shlex.quote(hybrid_quant_schema_path)} "
            f"--hybrid_quant_schema_re_path {shlex.quote(hybrid_quant_schema_re_path)} "
            f"--sensitivity_scores_save_path {shlex.quote(self.sensitivity_scores_save_path)} "
        )
        return cmd

    def run(
        self, run_id: int, budget_mb: int) -> str:
        """
        运行 AQT 获取敏感度分析得到的量化配置。
        """
        save_dir = os.path.join(self.results_root, f"run{run_id:03d}")
        os.makedirs(save_dir, exist_ok=True)
        hybrid_quant_schema_path = os.path.join(save_dir, "hybrid_quant_schema.json")
        hybrid_quant_schema_re_path = os.path.join(
            save_dir, "hybrid_quant_schema_re.json"
        )

        run_cmd = self._run_cmd(
            budget_mb,
            hybrid_quant_schema_path,
            hybrid_quant_schema_re_path,
        )
        success, stdout, stderr = ShellRunner.run_cmd(run_cmd, timeout=10800)

        if not success:
            logger.error("AQT failed to get hybrid quant schema.")

        return hybrid_quant_schema_path, hybrid_quant_schema_re_path
