import json
import os
import shlex

from ..utils.logger import logger
from ..utils.shell import ShellRunner


class AutomaticQuantizationTool:
    """
    封装 AQT 的两步调用：
    1) 计算敏感度，生成 mse_scores.json
    2) 基于模板 + 敏感度，输出最终的 modelslim YAML
    """

    def __init__(
        self,
        aqt_config: dict,
        tuning_strategy: dict,
        base_model_path: str,
        workspace: dict,
        quant_type: str = "minmax",
    ):
        self.config = aqt_config or {}
        self.tuning_strategy = tuning_strategy or {}
        self.base_model_path = os.path.abspath(base_model_path)
        self.workspace = workspace or {}
        self.quant_type = quant_type

        self.metric = self.config.get("sensitivity_metric", "mse")
        self.results_root = os.path.abspath(
            self.config.get("results_dir")
            or os.path.join(self.workspace.get("base_dir", "workspace"), "aqt_results")
        )
        self.omp_num_threads = int(self.config.get("omp_num_threads", 32))
        self.visible_devices = str(self.config.get("ascend_visible_devices", "0"))

        self.layer_configs_path = os.path.join(self.results_root, "layer_configs.json")
        self.initial_configs_rules_path = os.path.join(self.results_root, "initial_configs_rules.json")
        self.sensitivity_scores_save_path = os.path.join(
            self.results_root, self.quant_type, "sensitivity_scores.json"
        )
        self.priority_queue = []

        # Save rules to JSON for launch.py
        os.makedirs(self.results_root, exist_ok=True)
        with open(self.initial_configs_rules_path, "w") as f:
            json.dump(self.tuning_strategy.get("initial_layer_configs", []), f, indent=4)

    def get_priority_queue(self):
        """
        Calculates all possible upgrades for all layers based on custom upgrade_schemes 
        and sorts them by efficiency value.
        """
        if self.priority_queue:
            return self.priority_queue

        if not os.path.exists(self.sensitivity_scores_save_path):
            logger.error("Sensitivity scores not found. Call compute_sensitivity_scores_only first.")
            return []

        with open(self.sensitivity_scores_save_path, 'r') as f:
            scores = json.load(f)

        import re
        upgrade_rules = self.tuning_strategy.get("upgrade_schemes", [])
        
        # Default path if no rule matches
        default_path = ["w4a8_dynamic", "w8a8_default", "w8a8_dynamic", "float"]

        upgrades = []
        for name, data in scores.items():
            # 1. Determine upgrade path for this layer
            path = default_path
            for rule in upgrade_rules:
                reg = rule.get("regex")
                if reg and re.match(reg, name):
                    path = rule.get("path", default_path)
                    break
            
            # 2. Generate transitions (Phase = index in path)
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i+1]
                phase = i + 1
                
                # Skip invalid source configs for simulation
                if source not in ["w4a8_dynamic", "w8a8_default", "w8a8_dynamic"]:
                    continue
                
                val = self._calculate_transition_value(data, source, target)
                
                # Determine size increment for apply_upgrades logic if needed
                # (Though apply_upgrades currently doesn't use size_inc, only phase)
                upgrades.append({
                    "layer": name,
                    "target": target,
                    "source": source, # Helpful for validation
                    "val": val,
                    "phase": phase
                })

        # Phase ASC, score DESC
        self.priority_queue = sorted(upgrades, key=lambda x: (x['phase'], -x['val']))
        return self.priority_queue

    def _calculate_transition_value(self, data, source, target):
        """
        Helper to calculate accuracy gain for a transition.
        Now uses absolute metric reduction (e.g. MSE, Cosine) instead of size-normalized efficiency.
        """
        # score4 and score8 are now raw metric values from compute_sensitivity_scores.py
        val4 = data.get("score4", 0)
        val8 = data.get("score8", 0)
        val_f = 0.0
        
        # Helper to get Metric for a config name
        def get_val(cfg):
            if cfg == "w4a8_dynamic": return val4
            if cfg in ["w8a8_default", "w8a8_dynamic"]: return val8
            return val_f

        m_src = get_val(source)
        m_tgt = get_val(target)
        
        # Absolute Metric Reduction
        diff_metric = max(m_src - m_tgt, 0)
        
        # Special case: w8_default -> w8_dynamic (No weight metric change in simulation)
        # We still want to prioritize fixing sensitive layers first in Phase 1.
        if source == "w8a8_default" and target == "w8a8_dynamic":
            return val8 # Higher metric value means more sensitive
            
        return diff_metric

    def apply_upgrades(self, k: int):
        """
        Applies the next k valid upgrades from the priority queue to the current layer_configs state.
        Respects strict phasing: only upgrades from the lowest active phase are applied.
        """
        if not os.path.exists(self.layer_configs_path):
            logger.warning("No layer_configs.json found. Initial heuristic will be used in launch.py")
            return

        with open(self.layer_configs_path, 'r') as f:
            current_state = json.load(f)

        pq = self.get_priority_queue()

        # 1. Identify valid upgrades and the current active phase
        valid_upgrades = []
        active_phase = 99

        for upgrade in pq:
            layer = upgrade["layer"]
            target = upgrade["target"]
            source = upgrade["source"]
            phase = upgrade["phase"]

            if layer not in current_state:
                continue
            curr = current_state[layer]
            
            # Map current state bits/scope to config names for comparison
            curr_cfg_name = "float"
            if curr["weight_bits"] == 4:
                curr_cfg_name = "w4a8_dynamic"
            elif curr["weight_bits"] == 8:
                curr_cfg_name = "w8a8_dynamic" if curr["act_scope"] == "per_token" else "w8a8_default"

            # A transition is valid if the source matches current state
            if source == curr_cfg_name:
                valid_upgrades.append(upgrade)
                active_phase = min(active_phase, phase)

        if not valid_upgrades:
            logger.info("No more valid upgrades available in priority queue.")
            return

        # 2. Filter upgrades to only those in the current active phase
        target_pool = [u for u in valid_upgrades if u["phase"] == active_phase]
        logger.info(f"Tuning Phase {active_phase}: {len(target_pool)} potential upgrades remaining.")

        # 3. Apply top k upgrades from the target pool
        upgraded_count = 0
        for upgrade in target_pool:
            if upgraded_count >= k:
                break

            layer = upgrade["layer"]
            target = upgrade["target"]
            curr = current_state[layer]

            if target == "float":
                curr["weight_bits"] = 16
            elif target == "w8a8_default":
                curr["weight_bits"] = 8
                curr["act_scope"] = "per_tensor"
            elif target == "w8a8_dynamic":
                curr["weight_bits"] = 8
                curr["act_scope"] = "per_token"
            elif target == "w4a8_dynamic":
                curr["weight_bits"] = 4
                curr["act_scope"] = "per_token"

            logger.info(f"Applying Upgrade (Phase {active_phase}): {layer} -> {target}")
            upgraded_count += 1

        with open(self.layer_configs_path, 'w') as f:
            json.dump(current_state, f, indent=4)

        logger.info(f"Applied {upgraded_count} upgrades to state in Phase {active_phase}.")
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
            f"python -m src.aqt.compute_sensitivity_scores "
            f"--model-name-or-path {shlex.quote(self.base_model_path)} "
            f"--seed 42 "
            f"--quant-data-path {shlex.quote(quant_data_path)} "
            f"--quant-data-save-path {shlex.quote(quant_data_save_path)} "
            f"--quant-samples-num {self.config.get('quant_samples_num', 128)} "
            f"--quant-context-length {self.config.get('quant_context_length', 4096)} "
            f"--quant-type {self.quant_type} "
            f"--sensitivity-metric {shlex.quote(self.metric)} "
            f"--save-dir {shlex.quote(save_dir)} "
            f"--sensitivity_scores_save_path {shlex.quote(self.sensitivity_scores_save_path)} "
        )
        if self.config.get('is_mm', False):
            cmd += "--is-mm "
        if self.config.get('is_deepseek_v32', False):
            cmd += "--is-deepseek-v32 "
        return cmd

    def compute_sensitivity_scores_only(self, flag: bool = True):
        save_dir = os.path.join(self.results_root, self.quant_type)
        os.makedirs(save_dir, exist_ok=True)

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
                compute_sensitivity_scores_cmd,
                timeout=10800,
                log_path=os.path.join(self.results_root, "compute_sensitivity_scores.log")
            )
            if not success or not os.path.exists(self.sensitivity_scores_save_path):
                logger.error("Computing sensitivity scores failed.")
        else:
            logger.info("Sensitivity analysis skipped!")

    def _run_cmd(
        self,
        hybrid_quant_schema_path: str,
        hybrid_quant_schema_re_path: str,
    ) -> str:
        cmd = (
            f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(self.visible_devices)}; "
            f"export OMP_NUM_THREADS={self.omp_num_threads}; "
            f"python -m src.aqt.launch "
            f"--model-name-or-path {shlex.quote(self.base_model_path)} "
            f"--layer-configs-path {shlex.quote(self.layer_configs_path)} "
            f"--initial-configs-rules-path {shlex.quote(self.initial_configs_rules_path)} "
            f"--hybrid-quant-schema-path {shlex.quote(hybrid_quant_schema_path)} "
            f"--hybrid-quant-schema-re-path {shlex.quote(hybrid_quant_schema_re_path)} "
            f"--sensitivity-scores-save-path {shlex.quote(self.sensitivity_scores_save_path)} "
        )
        if self.config.get('is_mm', False):
            cmd += "--is-mm "
        return cmd

    def run(
        self, run_id: int) -> str:
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
            hybrid_quant_schema_path,
            hybrid_quant_schema_re_path,
        )
        success, stdout, stderr = ShellRunner.run_cmd(
            run_cmd,
            timeout=10800,
            log_path=os.path.join(self.results_root, "run.log")
        )

        if not success:
            logger.error("AQT failed to get hybrid quant schema.")

        return hybrid_quant_schema_path, hybrid_quant_schema_re_path
