import copy
import os
import yaml

_GLOBAL_CONFIG = None

DEFAULT_GENERATION_KWARGS = {
    "temperature": 0.5,
    "top_k": 10,
    "top_p": 0.95,
    "seed": None,
    "repetition_penalty": 1.03,
}

DEFAULT_VLLM_ARGS = {
    "trust-remote-code": True,
    "tensor-parallel-size": 2,
    "data-parallel-size": 1,
    "quantization": "ascend",
    "enable-prefix-caching": False,
    "max-model-len": 8192,
    "max-num-batched-tokens": 8192,
    "gpu-memory-utilization": 0.9,
    "additional_config": {
        "ascend_scheduler_config": {"enabled": True},
        "enable_weight_nz_layout": True,
    },
}


class GlobalConfig:
    def __init__(self):
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            self.user_config = yaml.safe_load(f) or {}
        self.raw_config = self._normalize_config(self.user_config)
        self.base_model_path = self.raw_config.get("base_model_path")
        if not self.base_model_path:
            raise ValueError("`base_model_path` must be provided in config/config.yaml.")

    def _normalize_config(self, cfg):
        normalized = {}
        normalized['base_model_path'] = cfg.get('base_model_path')

        normalized['workspace'] = {
            'base_dir': cfg.get('workspace_base_dir', 'workspace'),
            'current_run_dir': cfg.get('workspace_current_run_dir', 'current_run'),
            'history_dir': cfg.get('workspace_history_dir', 'history'),
            'best_weights_dir': cfg.get('workspace_best_weights_dir', 'best_weights'),
            'quant_config_name': cfg.get('workspace_quant_config_name', 'generated_modelslim_config.yaml'),
            'quant_weights_dir': cfg.get('workspace_quant_weights_dir', 'quantized_weights'),
        }

        normalized['strategy'] = {
            'layer_pattern': cfg.get('strategy_layer_pattern', 'model.layers.{i}.mlp.down_proj'),
            'initial_fallback_layers': cfg.get('strategy_initial_fallback_layers', ['lm_head']),
        }

        quant_template = cfg.get('quantization_template_config') or {}
        normalized['quantization'] = {
            'visible_devices': cfg.get('quantization_visible_devices', '0,1,2,3'),
            'model_type': cfg.get('quantization_model_type', 'Qwen3-32B'),
            'device': cfg.get('quantization_device', 'npu'),
            'trust_remote_code': cfg.get('quantization_trust_remote_code', True),
            'template_config': copy.deepcopy(quant_template),
        }
        if not normalized['quantization']['template_config']:
            raise ValueError("`quantization_template_config` must be provided in config/config.yaml.")

        evaluation = {
            'tolerance_ratio': cfg.get('evaluation_tolerance_ratio', 0.01),
            'datasets': cfg.get('evaluation_datasets') or cfg.get('datasets', {}),
            'disable_qwen_thinking': bool(cfg.get('disable_qwen_thinking', False)),
        }

        ais_generation = copy.deepcopy(DEFAULT_GENERATION_KWARGS)
        if isinstance(cfg.get('aisbench_generation_kwargs'), dict):
            ais_generation.update(cfg['aisbench_generation_kwargs'])

        # 是否使用 Chat 模版，决定 AISBench 使用的模型类型与配置前缀
        use_chat_template = bool(cfg.get('aisbench_use_chat_template', True))
        if use_chat_template:
            model_base_name = 'vllm_api_general_chat'
            model_abbr = 'vllm-api-general-chat'
            model_type = 'VLLMCustomAPIChat'
        else:
            model_base_name = 'vllm_api_general'
            model_abbr = 'vllm-api-general'
            model_type = 'VLLMCustomAPI'

        evaluation['aisbench'] = {
            'binary': 'ais_bench',
            'mode': 'all',
            'timeout': cfg.get('aisbench_timeout', 7200),
            'request_rate': cfg.get('aisbench_request_rate', 1),
            'retry': cfg.get('aisbench_retry', 2),
            'batch_size': cfg.get('aisbench_batch_size', 32),
            'max_out_len': cfg.get('aisbench_max_out_len', 512),
            'trust_remote_code': False,
            'pred_postprocessor': cfg.get('aisbench_pred_postprocessor', 'extract_non_reasoning_content'),
            'generation_kwargs': ais_generation,
            'model_config': {
                # 这几个字段不再暴露给用户，完全根据是否使用 Chat 模版自动推导
                'base_name': model_base_name,
                'abbr': model_abbr,
                'type': model_type,
                'attr': 'service',
                'subdir': 'vllm_api',
                'name_suffix': cfg.get('aisbench_model_name_suffix', 'auto'),
                'directory': cfg.get('aisbench_model_directory'),
                'use_chat_template': use_chat_template,
            },
            'log_dir': cfg.get('aisbench_log_dir', 'workspace/aisbench_logs'),
            'default_metric_keys': cfg.get(
                'aisbench_default_metric_keys',
                ['final_accuracy', 'accuracy', 'score']
            ),
            'extra_args': cfg.get('aisbench_extra_args'),
            'cleanup_model_config': cfg.get('aisbench_cleanup_model_config', True),
            'host_ip': cfg.get('aisbench_host_ip'),
            'host_port': cfg.get('aisbench_host_port'),
        }

        evaluation['datasets'] = evaluation['datasets'] or {}
        normalized['evaluation'] = evaluation

        vllm_args = copy.deepcopy(DEFAULT_VLLM_ARGS)
        if isinstance(cfg.get('vllm_args'), dict):
            vllm_args.update(cfg['vllm_args'])
        if 'served-model-name' not in vllm_args:
            served_model_name = cfg.get('vllm_served_model_name', 'qwen3')
            vllm_args['served-model-name'] = served_model_name

        normalized['vllm_server'] = {
            'entrypoint': 'vllm.entrypoints.openai.api_server',
            'env_vars': cfg.get('vllm_env_vars', {}),
            'host': 'localhost',
            'port': cfg.get('vllm_port', 1234),
            'health_check_endpoint': '/v1/models',
            'startup_timeout': 600,
            'args': vllm_args,
        }

        # Automatic Quantization Tool (AQT)
        default_aqt_results = os.path.join(normalized['workspace']['base_dir'], 'aqt_results')
        normalized['aqt'] = {
            'enabled': cfg.get('aqt_enabled', True),
            'project_dir': cfg.get('aqt_project_dir', 'automatic-quantization-tool'),
            'quant_data_path': cfg.get('aqt_quant_data_path'),
            'quant_data_save_path': cfg.get('aqt_quant_data_save_path'),
            'quant_samples_num': cfg.get('aqt_quant_samples_num', 128),
            'quant_context_length': cfg.get('aqt_quant_context_length', 4096),
            'sensitivity_metric': cfg.get('aqt_sensitivity_metric', 'mse'),
            'initial_budget_mb': cfg.get('aqt_initial_budget_mb', 2500),
            'budget_step_mb': cfg.get('aqt_budget_step_mb', 500),
            'budget_step_down_mb': cfg.get('aqt_budget_step_down_mb', 250),
            'min_budget_mb': cfg.get('aqt_min_budget_mb', 0),
            'max_budget_mb': cfg.get('aqt_max_budget_mb', 12000),
            'tighten_margin_ratio': cfg.get('aqt_tighten_margin_ratio', 0.01),
            'results_dir': cfg.get('aqt_results_dir', default_aqt_results),
            'omp_num_threads': cfg.get('aqt_omp_num_threads', 32),
            'ascend_visible_devices': cfg.get('aqt_ascend_visible_devices', "0"),
        }

        return normalized


def get_config():
    global _GLOBAL_CONFIG
    return _GLOBAL_CONFIG


def set_config():
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = GlobalConfig()


def set_global_args():
    set_config()
