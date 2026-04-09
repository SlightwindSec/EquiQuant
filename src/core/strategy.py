from ..utils.logger import logger


class TuningStrategy:
    def __init__(self, initial_fallback):
        """
        Args:
            initial_fallback (list): 来自 config 的初始回退列表
        """
        self.current_fallback = list(initial_fallback)
        self.history = []
        self.best_result = None
        logger.info(
            f"Strategy initialized. Fallback layers fixed: {len(self.current_fallback)}"
        )

    def next_trial(self, last_results, target_accuracies):
        """
        根据上次的精度字典，决定下一次的 fallback 层列表。

        Args:
            last_results (dict): e.g., {'ceval': 48.0, 'boolq': 52.0}
            target_accuracies (dict): e.g., {'ceval': 50.0, 'boolq': 65.0}

        Returns:
            list (fallback_layers) or str (STOP_SUCCESS / STOP_FAILED)
        """
        if last_results is None:
            # 第一次运行，使用初始 fallback 列表
            logger.info("First trial. Using initial fallback list.")
            return self.current_fallback

        # --- 评估上次结果 ---
        self.history.append(
            {"fallback": self.current_fallback, "results": last_results}
        )
        all_passed, failed_datasets = self.evaluate_results(
            last_results, target_accuracies
        )

        if all_passed:
            logger.info(
                f"All {len(target_accuracies)} datasets passed target accuracy!"
            )
            self.best_result = {
                "fallback": self.current_fallback,
                "results": last_results,
            }
            logger.info("Optimization goal reached. (Current strategy: STOP_SUCCESS)")
            return "STOP_SUCCESS"

        logger.error(
            f"Accuracy not met for: {failed_datasets}. No more fallback steps configured. (STOP_FAILED)"
        )
        return "STOP_FAILED"

    def evaluate_results(self, last_results, target_accuracies):
        """
        检查结果是否满足每个数据集的 target±tolerance。
        Returns:
            (bool all_passed, list failed_dataset_names)
        """
        if last_results is None:
            return False, list(target_accuracies.keys())

        all_passed = True
        failed = []

        for name, target_cfg in target_accuracies.items():
            value = last_results.get(name)
            lower, upper = self._calculate_bounds(target_cfg)

            if value is None:
                logger.error(f"Missing accuracy result for dataset '{name}'.")
                all_passed = False
                failed.append(name)
                continue

            if not (lower <= value <= upper):
                target_value = (upper + lower) / 2
                tolerance_pct = (
                    ((upper - lower) / 2) / target_value if target_value else 0.0
                )
                logger.warning(
                    f"[Dataset: {name}] accuracy={value:.4f}, "
                    f"target={target_value:.4f}±{tolerance_pct * 100:.2f}% "
                    f"(allowed range {lower:.4f}~{upper:.4f})"
                )
                all_passed = False
                failed.append(name)

        return all_passed, failed

    def _calculate_bounds(self, target_cfg):
        """
        target_cfg 可以是单个浮点数或包含 target_accuracy / tolerance_ratio 的 dict
        """
        if isinstance(target_cfg, dict):
            target = target_cfg.get("target_accuracy") or target_cfg.get("target")
            tolerance = target_cfg.get("tolerance_ratio") or target_cfg.get(
                "tolerance", 0.01
            )
        else:
            target = target_cfg
            tolerance = 0.0

        if target is None:
            raise ValueError("Target accuracy must be provided for each dataset.")

        lower = target * (1 - tolerance)
        upper = target * (1 + tolerance)
        return lower, upper
