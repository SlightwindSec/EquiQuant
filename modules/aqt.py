import os
import shlex
from typing import Optional, Tuple

from utils.logger import logger
from utils.shell import ShellRunner


class AutomaticQuantizationTool:
    """
    封装 AQT 的两步调用：
    1) 计算敏感度，生成 mse_scores.json
    2) 基于模板 + 敏感度，输出最终的 modelslim YAML
    """

    def __init__(self, aqt_config: dict, base_model_path: str, workspace: dict):
        self.config = aqt_config or {}
        self.base_model_path = base_model_path
        self.workspace = workspace or {}

        self.project_dir = os.path.abspath(self.config.get('project_dir', 'automatic-quantization-tool'))
        self.scripts_dir = os.path.join(self.project_dir, "scripts")
        self.metric = self.config.get('sensitivity_metric', 'mse')
        self.results_root = os.path.abspath(
            self.config.get('results_dir') or os.path.join(self.workspace.get('base_dir', 'workspace'), "aqt_results")
        )
        self.initial_budget = int(self.config.get('initial_budget_mb', 2500))
        self.max_budget = int(self.config.get('max_budget_mb', self.initial_budget))
        self.min_budget = int(self.config.get('min_budget_mb', 0))
        self.step_up = int(self.config.get('budget_step_mb', 500))
        self.step_down = int(self.config.get('budget_step_down_mb', self.step_up))
        self.tighten_margin_ratio = float(self.config.get('tighten_margin_ratio', 0.01))
        self.omp_num_threads = int(self.config.get('omp_num_threads', 32))
        self.visible_devices = str(self.config.get('ascend_visible_devices', "0"))

    # ------------------------------------------------------------------ #
    # 状态检查与预算调整
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        """检测当前目录是否存在 AQT 项目。"""
        if not os.path.isdir(self.project_dir):
            logger.info(f"AQT project directory not found: {self.project_dir}")
            return False
        if not os.path.isdir(self.scripts_dir):
            logger.info(f"AQT scripts directory not found: {self.scripts_dir}")
            return False
        return True

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

    def _sensitivity_scores_path(self, save_dir: str) -> str:
        return os.path.join(save_dir, "modelslim", self.metric, f"{self.metric}_scores.json")

    def _analysis_cmd(
        self,
        save_dir: str,
        budget_mb: int,
        quant_data_path: str,
        quant_data_save_path: str,
    ) -> str:
        project_dir_q = shlex.quote(self.project_dir)
        scripts_dir_q = shlex.quote(self.scripts_dir)
        cmd = (
            f"cd {scripts_dir_q} && "
            f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(self.visible_devices)}; "
            f"export OMP_NUM_THREADS={self.omp_num_threads}; "
            f"python {project_dir_q}/quantize_modelslim.py "
            f"--model-name-or-path {shlex.quote(self.base_model_path)} "
            f"--quant-data-path {shlex.quote(quant_data_path)} "
            f"--quant-data-save-path {shlex.quote(quant_data_save_path)} "
            f"--quant-samples-num {self.config.get('quant_samples_num', 128)} "
            f"--quant-context-length {self.config.get('quant_context_length', 4096)} "
            f"--quant-type modelslim "
            f"--quant-sym "
            f"--disable-smoothquant "
            f"--sensitivity-metric {shlex.quote(self.metric)} "
            f"--compute-sensitivity-scores-only "
            f"--ckpt-size-budget-mb {budget_mb} "
            f"--save-dir {shlex.quote(save_dir)} "
            f"--eval-ppl"
        )
        return cmd

    def _prepare_config_cmd(
        self,
        budget_mb: int,
        sensitivity_scores_path: str,
        template_path: str,
        output_path: str,
    ) -> str:
        project_dir_q = shlex.quote(self.project_dir)
        scripts_dir_q = shlex.quote(self.scripts_dir)
        cmd = (
            f"cd {scripts_dir_q} && "
            f"python {project_dir_q}/prepare_modelslim_config.py "
            f"--model-name-or-path {shlex.quote(self.base_model_path)} "
            f"--sensitivity-metric {shlex.quote(self.metric)} "
            f"--ckpt-size-budget-mb {budget_mb} "
            f"--sensitivity-scores-path {shlex.quote(sensitivity_scores_path)} "
            f"--template-path {shlex.quote(template_path)} "
            f"--output-path {shlex.quote(output_path)}"
        )
        return cmd

    def generate_modelslim_config(
        self,
        run_id: int,
        base_config_path: str,
        budget_mb: int,
        output_config_path: str,
    ) -> Optional[str]:
        """
        运行 AQT 两步，返回最终的 modelslim 配置路径。
        """
        quant_data_path = self.config.get('quant_data_path')
        if not quant_data_path:
            logger.error("AQT requires `aqt_quant_data_path` in config.")
            return None

        if not os.path.exists(base_config_path):
            logger.error(f"Base modelslim config for AQT not found: {base_config_path}")
            return None

        save_dir = self._build_save_dir(run_id, budget_mb)
        quant_data_save_path = self.config.get('quant_data_save_path') or os.path.join(save_dir, "calib_data.pt")
        os.makedirs(os.path.dirname(quant_data_save_path), exist_ok=True)

        # Step 1: 敏感度分析
        analysis_cmd = self._analysis_cmd(save_dir, budget_mb, quant_data_path, quant_data_save_path)
        success, stdout, stderr = ShellRunner.run_cmd(analysis_cmd, timeout=10800)
        if not success:
            logger.error("AQT sensitivity analysis failed.")
            return None

        sensitivity_scores_path = self._sensitivity_scores_path(save_dir)
        if not os.path.exists(sensitivity_scores_path):
            logger.error(f"AQT sensitivity scores not found: {sensitivity_scores_path}")
            return None

        # Step 2: 生成最终 YAML
        prepare_cmd = self._prepare_config_cmd(budget_mb, sensitivity_scores_path, base_config_path, output_config_path)
        success, stdout, stderr = ShellRunner.run_cmd(prepare_cmd, timeout=3600)
        if not success:
            logger.error("AQT prepare_modelslim_config failed.")
            return None

        if not os.path.exists(output_config_path):
            logger.error(f"AQT output config missing after prepare step: {output_config_path}")
            return None

        return output_config_path

