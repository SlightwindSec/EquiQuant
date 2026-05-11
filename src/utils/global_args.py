import copy
import os
import yaml


class GlobalConfig:
    """
    GlobalConfig is a singleton class that holds the global configuration for the EquiQuant engine.
    """
    _instance = None
    _initialized = False

    SUPPORTED_QUANTIZATION_TOOLS = {"llmcompressor", "msmodelslim"}
    _DEFAULT_MODEL_CONFIG = {"is_mm": False, "is_deepseek_v32": False}
    _DEFAULT_VISIBLE_DEVICE = "0"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file_path: str, quantization_tool: str):
        if self._initialized:
            return
        self._initialized = True

        with open(config_file_path, "r", encoding="utf-8") as f:
            self._user_config = yaml.safe_load(f) or {}
        self.raw_config = {}
        self._normalize_config(quantization_tool)

    @classmethod
    def get_instance(cls) -> "GlobalConfig":
        return self._instance

    def _normalize_config(self, quantization_tool: str):
        """
        Normalize the configuration. Result is stored in self.raw_config.
        """
        base_model_path = self._user_config.get("base_model_path")
        if not base_model_path:
            raise ValueError("`base_model_path` must be provided in config_file_path")
        self.raw_config["base_model_path"] = base_model_path

        self._normalize_workspace_config()
        self._normalize_strategy_config()
        self._normalize_aqt_config()
        self._normalize_quantization_config(quantization_tool)
        self._normalize_vllm_config()
        self._normalize_evaluation_config()
    
    def _normalize_workspace_config(self) -> None:
        """
        Normalize the workspace configuration.
        """
        self.raw_config["workspace"] = {
            "base_dir": self._user_config.get("workspace_base_dir", "workspace"),
            "current_run_dir": self._user_config.get("workspace_current_run_dir", "current_run"),
            "best_weights_dir": self._user_config.get("workspace_best_weights_dir", "best_weights"),
            "quant_weights_dir": self._user_config.get(
                "workspace_quant_weights_dir", "quantized_weights"
            ),
        }
    
    def _normalize_strategy_config(self) -> None:
        self.raw_config["strategy"] = {
            "initial_fallback_layers":  self._user_config.get(
                "strategy_initial_fallback_layers", ["lm_head"]
            ),
        }
        self.raw_config["disable_names"] =  self._user_config.get("disable_names", [])

    def _normalize_aqt_config(self) -> None:
        """
        Normalize the AQT configuration.
        """
        if self._user_config.get("aqt_quant_data_path") is None:
            raise ValueError("aqt_quant_data_path is required")

        default_aqt_results = os.path.join(
            self._user_config.get("workspace_base_dir", "workspace"), "aqt_results"
        )
        self.raw_config["aqt"] = {
            "quant_data_path": self._user_config.get("aqt_quant_data_path"),
            "is_mm": self._user_config.get("is_mm", self._DEFAULT_MODEL_CONFIG["is_mm"]),
            "is_deepseek_v32": self._user_config.get(
                "is_deepseek_v32", self._DEFAULT_MODEL_CONFIG["is_deepseek_v32"]
                ),
            "results_dir": self._user_config.get("aqt_results_dir", default_aqt_results),
            "omp_num_threads": self._user_config.get("aqt_omp_num_threads", 32),
            "ascend_visible_devices": self._user_config.get(
                "aqt_ascend_visible_devices", self._DEFAULT_VISIBLE_DEVICE
                ),
            "quant_samples_num": self._user_config.get("aqt_quant_samples_num", 128),
            "quant_context_length": self._user_config.get("aqt_quant_context_length", 4096),
            "sensitivity_metrics": self._user_config.get("aqt_sensitivity_metrics", ["mse"]),
            "initial_budget_mb": self._user_config.get("aqt_initial_budget_mb", 2500),
            "budget_step_mb": self._user_config.get("aqt_budget_step_mb", 500),
            "budget_step_down_mb": self._user_config.get("aqt_budget_step_down_mb", 250),
            "min_budget_mb": self._user_config.get("aqt_min_budget_mb", 0),
            "max_budget_mb": self._user_config.get("aqt_max_budget_mb", 12000),
        }

    def _normalize_quantization_config(self, quantization_tool: str) -> None:
        """
        Normalize the quantization configuration.
        """

        # checks if quantization_tool is supported, then store it if supported
        if quantization_tool not in self.SUPPORTED_QUANTIZATION_TOOLS:
            raise ValueError(
                f"`quantization_tool` should be one of {self.SUPPORTED_QUANTIZATION_TOOLS}, but got '{quantization_tool}'"
            )
        self.raw_config["quantization_tool"] = quantization_tool

        # if quantization_tool specific device/data is not provided, use aqt's instead
        visible_devices = self._user_config.get(
            "quantization_visible_devices",
            self._DEFAULT_VISIBLE_DEVICE
            )
        calib_data_path = self._user_config.get(
            "quantization_calib_data_path",
            self._user_config.get("aqt_quant_data_path")
            )

        # llmcompressor only supports single-cardPU quantization
        if quantization_tool == "llmcompressor" and "," in str(visible_devices).strip():
            visible_devices = visible_devices.split(",")[0]
            print(f"llmcompressor only supports single-cardPU quantization, using '{visible_devices}' instead.")

        # normalize quant config based on quantization tool chosen
        if quantization_tool == "msmodelslim":
            self._normalize_msmodelslim_config()
        elif quantization_tool == "llmcompressor":
            self._normalize_llmcompressor_config(calib_data_path)
        
        self.raw_config["quantization"].update({
            "visible_devices": visible_devices,
            "model_type": self._user_config.get("quantization_model_type", "Qwen3-32B"),
            "device": self._user_config.get("quantization_device", "npu"),
            })
    
    def _normalize_msmodelslim_config(self) -> None:
        """
        Normalize the msmodelslim configuration.
        """
        # 优先从quantization_template_config中读取w_bit/a_bit，如果没有则从旧字段读取
        quant_template = self._user_config.get("quantization_template_config")
        if not quant_template:
            raise ValueError(
                "`quantization_template_config` must be provided in config/config.yaml for msmodelslim."
            )

        self.raw_config["quantization"] = {
            "is_mm": self._user_config.get("is_mm", self._DEFAULT_MODEL_CONFIG["is_mm"]),
            "is_deepseek_v32": self._user_config.get(
                "is_deepseek_v32", self._DEFAULT_MODEL_CONFIG["is_deepseek_v32"]
                ),
            "trust_remote_code": self._user_config.get("quantization_trust_remote_code", True),
            "template_config": copy.deepcopy(quant_template),
            "w_bit": quant_template.get("w_bit", 4),
            "a_bit": quant_template.get("a_bit", 8),
        }
    
    def _normalize_llmcompressor_config(self, calib_data_path: str) -> None:
        """
        Normalize the llmcompressor configuration.
        """
        self.raw_config["quantization"] = {
            "enable_smooth_quant": self._user_config.get("enable_smooth_quant", False),
            "smooth_strength": self._user_config.get("smooth_strength", 0.8),
            "calib_data_path": calib_data_path,
            "num_calibration_samples": self._user_config.get("num_calibration_samples", 512),
            "max_sequence_length": self._user_config.get("max_sequence_length", 2048),
            "modifier": self._user_config.get("quantization_modifier", "PTQ"),
        }
    
    def _normalize_vllm_config(self) -> None:
        """
        Normalize the vllm configuration.
        """
        DEFAULT_VLLM_ARGS = {
            "trust-remote-code": True,
            "tensor-parallel-size": 2,
            "data-parallel-size": 1,
            "enable-prefix-caching": False,
            "max-model-len": 8192,
            "max-num-batched-tokens": 8192,
            "gpu-memory-utilization": 0.9,
            "additional-config": {
                "ascend_scheduler_config": {"enabled": False},
                "enable_cpu_binding": True,
                "enable_weight_nz_layout": True,
            },
        }
        vllm_args = copy.deepcopy(DEFAULT_VLLM_ARGS)
        if isinstance(self._user_config.get("vllm_args"), dict):
            vllm_args.update(self._user_config["vllm_args"])
        if self._user_config.get("quantization_tool") == "msmodelslim":
            vllm_args["quantization"] = "ascend"
        if "served-model-name" not in vllm_args:
            served_model_name = self._user_config.get("vllm_served_model_name", "model")
            vllm_args["served-model-name"] = served_model_name

        self.raw_config["vllm_server"] = {
            "entrypoint": "vllm.entrypoints.openai.api_server",
            "env_vars": self._user_config.get("vllm_env_vars", {}),
            "host": "localhost",
            "port": self._user_config.get("vllm_port", 1234),
            "health_check_endpoint": "/v1/models",
            "startup_timeout": 1800,
            "args": vllm_args,
        }

    def _normalize_evaluation_config(self) -> None:
        """
        Normalize the evaluation configuration.
        """
        DEFAULT_GENERATION_KWARGS = {
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "seed": None,
            "repetition_penalty": 1.03,
        }

        evaluation = {
            "tolerance_ratio": self._user_config.get("evaluation_tolerance_ratio", 1.0),
            "datasets": self._user_config.get("evaluation_datasets", {}),
            "disable_qwen_thinking": self._user_config.get("disable_qwen_thinking", False),
        }

        ais_generation = copy.deepcopy(DEFAULT_GENERATION_KWARGS)
        if isinstance(self._user_config.get("aisbench_generation_kwargs"), dict):
            ais_generation.update(self._user_config["aisbench_generation_kwargs"])

        # 是否使用 Chat 模版，决定 AISBench 使用的模型类型与配置前缀
        use_chat_template = self._user_config.get("aisbench_use_chat_template", True)
        if use_chat_template:
            model_base_name = "vllm_api_general_chat"
            model_abbr = "vllm-api-general-chat"
            model_type = "VLLMCustomAPIChat"
        else:
            model_base_name = "vllm_api_general"
            model_abbr = "vllm-api-general"
            model_type = "VLLMCustomAPI"   

        default_evaluation_results = os.path.join(
            self._user_config.get("workspace_base_dir", "workspace"), "aisbench_logs"
        )
        evaluation["aisbench"] = {
            "binary": "ais_bench",
            "mode": "all",
            "timeout": self._user_config.get("aisbench_timeout", 7200),
            "request_rate": self._user_config.get("aisbench_request_rate", 1),
            "retry": self._user_config.get("aisbench_retry", 2),
            "batch_size": self._user_config.get("aisbench_batch_size", 32),
            "max_out_len": self._user_config.get("aisbench_max_out_len", 512),
            "trust_remote_code": False,
            "pred_postprocessor": self._user_config.get(
                "aisbench_pred_postprocessor", "extract_non_reasoning_content"
            ),
            "generation_kwargs": ais_generation,
            "model_config": {
                # 这几个字段不再暴露给用户，完全根据是否使用 Chat 模版自动推导
                "base_name": model_base_name,
                "abbr": model_abbr,
                "type": model_type,
                "attr": "service",
                "subdir": "vllm_api",
                "name_suffix": self._user_config.get("aisbench_model_name_suffix", "auto"),
                "directory": self._user_config.get("aisbench_model_directory"),
                "use_chat_template": use_chat_template,
            },
            "log_dir": default_evaluation_results,
            "default_metric_keys": self._user_config.get(
                "aisbench_default_metric_keys", ["final_accuracy", "accuracy", "score"]
            ),
            "extra_args": self._user_config.get("aisbench_extra_args"),
            "cleanup_model_config": self._user_config.get("aisbench_cleanup_model_config", True),
            "host_ip": self._user_config.get("aisbench_host_ip"),
            "host_port": self._user_config.get("aisbench_host_port"),
        }

        self.raw_config["evaluation"] = evaluation
