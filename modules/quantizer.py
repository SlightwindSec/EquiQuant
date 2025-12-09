import os
import copy
import importlib
from utils.shell import ShellRunner
from utils.file_io import write_yaml
from utils.logger import logger


class ModelslimQuantizer:
    """
    封装 Modelslim 量化工具。
    """
    def __init__(
        self,
        quant_config,
        base_model_path,
        fallback_layers,
        output_config_path,
        output_weights_path,
        prepared_config_path=None,
    ):
        """
        Args:
            quant_config (dict): 来自 config.yaml['quantization']
            base_model_path (str): 来自 config.yaml['base_model_path']
            fallback_layers (list): 本次回退层列表
            output_config_path (str): 本次运行 modelslim YAML 保存路径
            output_weights_path (str): 本次运行量化权重输出路径
            prepared_config_path (str, optional): 已经生成好的 modelslim 配置路径（AQT 场景使用）
        """
        self.config = quant_config
        self.base_model_path = base_model_path
        self.fallback_layers = fallback_layers
        self.output_config_path = output_config_path
        self.output_weights_path = output_weights_path
        self.prepared_config_path = prepared_config_path

    def _resolve_calib_dataset_path(self, dataset_name_or_path):
        """
        解析标定数据集的最终路径。
        
        - 如果是绝对路径 (e.g., /home/...), 直接使用。
        - 如果是相对路径/文件名 (e.g., mix_calib.jsonl), 
          则在 msmodelslim 包的 'lab_calib' 目录中查找。
        """
        if os.path.isabs(dataset_name_or_path):
            logger.info(f"Using user-provided absolute path for calib_dataset: {dataset_name_or_path}")
            return dataset_name_or_path

        # 不是绝对路径，假定为默认数据集，需要在 msmodelslim 包中查找
        logger.info(f"Resolving default dataset '{dataset_name_or_path}' from msmodelslim package...")
        try:
            # 导入 msmodelslim 包
            spec = importlib.util.find_spec("msmodelslim")
            if spec is None or spec.origin is None:
                raise ImportError("Cannot find 'msmodelslim' package. Is it installed?")

            # spec.origin 通常是 .../msmodelslim/__init__.py
            # 需要 .../msmodelslim/
            package_dir = os.path.dirname(spec.origin)
            resolved_path = os.path.join(package_dir, "lab_calib", dataset_name_or_path)

            if not os.path.exists(resolved_path):
                logger.warning(f"Resolved path does not exist: {resolved_path}")
                raise FileNotFoundError(f"Could not find default dataset at {resolved_path}")

            logger.info(f"Resolved default dataset path to: {resolved_path}")
            return resolved_path

        except Exception as e:
            logger.error(f"Failed to resolve calib_dataset path for '{dataset_name_or_path}': {e}")
            raise

    def _generate_quant_config(self, disable_names=None, output_path=None):
        """
        从 config.yaml 中的模板动态生成 modelslim 的配置文件。
        """
        logger.debug("Generating modelslim config file...")

        try:
            config_data = copy.deepcopy(self.config['template_config'])

            # 1. 插入动态的回退层列表（可选，AQT 模式下可以为 None 以便由 AQT 决定）
            if disable_names is None:
                disable_names = self.fallback_layers
            if disable_names is not None:
                config_data['spec']['calib_cfg']['disable_names'] = disable_names

            # 2. 解析标定数据集路径
            calib_name = config_data['spec']['calib_dataset']
            calib_full_path = self._resolve_calib_dataset_path(calib_name)
            config_data['spec']['calib_dataset'] = "mix_calib.jsonl"

            # 3. 确保 metadata 中的 model_type 与命令行一致
            config_data['metadata']['verified_model_types'] = [self.config['model_type']]

            # 4. 将配置写入临时文件
            if output_path:
                self.output_config_path = output_path
            if not write_yaml(config_data, self.output_config_path):
                raise Exception("Failed to write dynamic config file.")

            return self.output_config_path

        except KeyError as e:
            logger.error(f"Config Error: 'config.yaml' 中 'quantization.template_config' 缺少键: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate quant config: {e}")
            raise

    def generate_config_only(self, disable_names=None, output_path=None):
        """
        仅生成 modelslim 配置文件（不执行量化），用于 AQT 前置步骤。
        """
        try:
            return self._generate_quant_config(disable_names=disable_names, output_path=output_path)
        except Exception as e:
            logger.error(f"Failed to generate quant config only: {e}")
            return None

    def run(self):
        """
        执行完整的量化流程。
        """
        logger.info(f"Starting quantization... Fallback layers: {len(self.fallback_layers)}")
        try:
            if self.prepared_config_path:
                dynamic_config_path = self.prepared_config_path
                logger.info(f"Using pre-generated quant config: {dynamic_config_path}")
            else:
                dynamic_config_path = self._generate_quant_config()
            env_prefix = f"export ASCEND_RT_VISIBLE_DEVICES={self.config['visible_devices']}; "
            cmd = (
                f"msmodelslim quant "
                f"--model_path {self.base_model_path} " 
                f"--save_path {self.output_weights_path} "
                f"--config_path {dynamic_config_path} "
                f"--device {self.config['device']} "
                f"--model_type {self.config['model_type']} "
                f"--trust_remote_code {self.config['trust_remote_code']}"
            )
            full_cmd = env_prefix + cmd
            success, stdout, stderr = ShellRunner.run_cmd(full_cmd, timeout=10800)
            if not success:
                logger.error(f"Modelslim quantization failed. Stderr: {stderr}")
                raise Exception(f"Modelslim failed.")
            logger.info("Quantization finished successfully.")
            return self.output_weights_path
        except Exception as e:
            logger.error(f"An error occurred during quantization run: {e}")
            return None
