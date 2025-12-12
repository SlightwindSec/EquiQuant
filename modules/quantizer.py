import os
import copy
from utils.shell import ShellRunner
from utils.file_io import write_yaml
from utils.logger import logger

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    RUAMEL_AVAILABLE = True
except ImportError:
    RUAMEL_AVAILABLE = False
    logger.warning("ruamel.yaml not available. v1 format support may be limited.")


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

    def _generate_v0_config(self, disable_names=None, output_path=None):
        """
        生成 modelslim v0 格式的配置文件（legacy格式）。
        """
        logger.debug("Generating modelslim v0 config file...")
        
        try:
            template = self.config['template_config']
            v0_config = template.get('v0', {})
            if not v0_config:
                config_data = copy.deepcopy(template)
            else:
                config_data = copy.deepcopy(v0_config)
                config_data['apiversion'] = 'modelslim_v0'
            # 统一写入全局量化 bit 配置
            w_bit = self.config.get('w_bit')
            a_bit = self.config.get('a_bit')
            label = config_data.setdefault('metadata', {}).setdefault('label', {})
            if w_bit is not None:
                label['w_bit'] = w_bit
            if a_bit is not None:
                label['a_bit'] = a_bit

            calib_cfg = config_data.setdefault('spec', {}).setdefault('calib_cfg', {})
            if w_bit is not None:
                calib_cfg['w_bit'] = w_bit
            if a_bit is not None:
                calib_cfg['a_bit'] = a_bit

            if disable_names is None:
                disable_names = self.fallback_layers
            if disable_names is not None:
                config_data['spec']['calib_cfg']['disable_names'] = disable_names

            config_data['spec']['calib_dataset'] = "mix_calib.jsonl"
            config_data['metadata']['verified_model_types'] = [self.config['model_type']]

            if output_path:
                self.output_config_path = output_path
            if not write_yaml(config_data, self.output_config_path):
                raise Exception("Failed to write dynamic config file.")

            return self.output_config_path

        except KeyError as e:
            logger.error(f"Config Error: 'config.yaml' 中 'quantization.template_config' 缺少键: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate v0 quant config: {e}")
            raise

    def _generate_v1_config(self, disable_names=None, output_path=None):
        """
        生成 modelslim v1 格式的配置文件（使用YAML锚点）。
        """
        if not RUAMEL_AVAILABLE:
            raise ImportError("ruamel.yaml is required for v1 format. Please install it: pip install ruamel.yaml")

        logger.debug("Generating modelslim v1 config file...")

        try:
            template = self.config['template_config']
            v1_config = template.get('v1', {})

            # 兼容旧格式：如果没有v1配置，尝试从旧格式读取
            if not v1_config:
                # 兼容旧格式：从顶层读取v1_qconfigs等
                v1_config = {
                    'qconfigs': template.get('v1_qconfigs', {}),
                    'process': template.get('v1_process', []),
                    'save': template.get('v1_save', []),
                    'metadata': template.get('metadata', {})
                }

            yaml = YAML()
            yaml.preserve_quotes = True
            yaml.width = 4096  # 避免长行被截断

            def _should_ignore_aliases(data):
                """
                只为显式设置了锚点的节点保留 alias，避免标量被自动抽象出锚点。
                """
                if hasattr(data, "yaml_anchor"):
                    anchor = data.yaml_anchor()
                    if anchor and anchor.always_dump:
                        return False
                return True

            yaml.representer.ignore_aliases = _should_ignore_aliases

            # 构建根配置
            root = CommentedMap()
            root['apiversion'] = 'modelslim_v1'

            # metadata部分
            metadata = CommentedMap()
            v1_metadata = v1_config.get('metadata', {})
            metadata['config_id'] = v1_metadata.get('config_id', 'qwen3-dense-w8a8-v1')
            metadata['score'] = v1_metadata.get('score', 90)
            verified_model_types = v1_metadata.get('verified_model_types')
            if verified_model_types:
                metadata['verified_model_types'] = list(verified_model_types)
            else:
                metadata['verified_model_types'] = [self.config['model_type']]

            # label部分
            label = CommentedMap()
            w_bit = self.config.get('w_bit', 8)
            a_bit = self.config.get('a_bit', 8)
            label['w_bit'] = w_bit
            label['a_bit'] = a_bit
            v1_label = v1_metadata.get('label', {})
            label['is_sparse'] = v1_label.get('is_sparse', False)
            label['kv_cache'] = v1_label.get('kv_cache', False)
            metadata['label'] = label
            root['metadata'] = metadata

            # 定义qconfig锚点（从模板的v1.qconfigs中读取）
            # 生成所有在qconfigs中定义的qconfig锚点，同时缓存节点供别处引用
            v1_qconfigs = v1_config.get('qconfigs', {})
            qconfig_nodes = {}

            def _add_qconfig(name, act_cfg, weight_cfg):
                qconfig = CommentedMap()
                act = CommentedMap()
                act['scope'] = act_cfg.get('scope', 'per_token')
                act['dtype'] = act_cfg.get('dtype', 'int8')
                act['symmetric'] = act_cfg.get('symmetric', True)
                act['method'] = act_cfg.get('method', 'minmax')
                qconfig['act'] = act

                weight = CommentedMap()
                weight['scope'] = weight_cfg.get('scope', 'per_channel')
                weight['dtype'] = weight_cfg.get('dtype', 'int8')
                weight['symmetric'] = weight_cfg.get('symmetric', True)
                weight['method'] = weight_cfg.get('method', 'minmax')
                if 'ext' in weight_cfg:
                    weight['ext'] = CommentedMap(weight_cfg['ext'])
                qconfig['weight'] = weight

                qconfig.yaml_set_anchor(name, always_dump=True)
                qconfig_nodes[name] = qconfig
                root[name] = qconfig

            # 如果没有配置v1_qconfigs，根据w_bit和a_bit生成默认配置（同时补齐 w8a8 和 w4a8 两套）
            if not v1_qconfigs:
                _add_qconfig(
                    'default_w8a8_dynamic',
                    {'scope': 'per_token', 'dtype': 'int8', 'symmetric': True, 'method': 'minmax'},
                    {'scope': 'per_channel', 'dtype': 'int8', 'symmetric': True, 'method': 'minmax'},
                )
                _add_qconfig(
                    'default_w4a8_dynamic',
                    {'scope': 'per_token', 'dtype': 'int8', 'symmetric': True, 'method': 'minmax'},
                    {'scope': 'per_group', 'dtype': 'int4', 'symmetric': True, 'method': 'minmax', 'ext': {'group_size': 64}},
                )
            else:
                # 生成所有在v1_qconfigs中定义的qconfig锚点
                for qconfig_name, qconfig_template in v1_qconfigs.items():
                    _add_qconfig(
                        qconfig_name,
                        qconfig_template.get('act', {}),
                        qconfig_template.get('weight', {}),
                    )

            # 根据w_bit选择合适的默认qconfig名称（用于process配置中），如果不存在则使用第一个
            if w_bit == 4:
                default_qconfig_name = 'default_w4a8_dynamic'
            else:
                default_qconfig_name = 'default_w8a8_dynamic'

            if default_qconfig_name not in qconfig_nodes and qconfig_nodes:
                default_qconfig_name = list(qconfig_nodes.keys())[0]

            # spec部分
            spec = CommentedMap()

            # process配置
            v1_process = v1_config.get('process', [])
            if not v1_process:
                # 如果没有配置，使用默认的process配置
                process_group = CommentedMap()
                process_group['type'] = 'group'
                configs = []

                def _append_linear_quant(qconfig_name):
                    linear_quant = CommentedMap()
                    linear_quant['type'] = 'linear_quant'
                    linear_quant['qconfig'] = qconfig_nodes[qconfig_name]
                    linear_quant['include'] = []
                    linear_quant['exclude'] = []
                    configs.append(linear_quant)

                # 默认情况下优先输出 w4a8 -> w8a8 的顺序，如果存在的话
                preferred_order = ['default_w4a8_dynamic', 'default_w8a8_dynamic']
                existing_order = [name for name in preferred_order if name in qconfig_nodes]
                if not existing_order:
                    existing_order = list(qconfig_nodes.keys())

                for name in existing_order:
                    _append_linear_quant(name)

                process_group['configs'] = configs
                spec['process'] = [process_group]
            else:
                # 使用模板中的process配置，但需要处理锚点引用
                process_list = []
                for proc_item in v1_process:
                    proc_map = CommentedMap(proc_item)
                    if 'configs' in proc_map:
                        configs_list = []
                        for cfg_item in proc_map['configs']:
                            cfg_map = CommentedMap(cfg_item)
                            # 如果qconfig_ref存在，转换为锚点引用
                            if 'qconfig_ref' in cfg_map:
                                ref_name = cfg_map.pop('qconfig_ref')
                                ref_qconfig = qconfig_nodes.get(ref_name)
                                if ref_qconfig is None:
                                    raise ValueError(f"qconfig_ref '{ref_name}' not found in defined qconfigs")
                                cfg_map['qconfig'] = ref_qconfig
                            # include/exclude不需要用户配置，AQT会自动填充，但生成时留空
                            if 'include' not in cfg_map:
                                cfg_map['include'] = []
                            if 'exclude' not in cfg_map:
                                cfg_map['exclude'] = []
                            configs_list.append(cfg_map)
                        proc_map['configs'] = configs_list
                    process_list.append(proc_map)
                spec['process'] = process_list

            # save配置
            v1_save = v1_config.get('save', [])
            if not v1_save:
                # 默认save配置
                save_item = CommentedMap()
                save_item['type'] = 'ascendv1_saver'
                save_item['part_file_size'] = 4
                spec['save'] = [save_item]
            else:
                spec['save'] = [CommentedMap(item) for item in v1_save]

            root['spec'] = spec

            if output_path:
                self.output_config_path = output_path

            directory = os.path.dirname(self.output_config_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            with open(self.output_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(root, f)

            logger.info(f"Successfully generated v1 config file: {self.output_config_path}")
            return self.output_config_path

        except KeyError as e:
            logger.error(f"Config Error: 'config.yaml' 中 'quantization.template_config' 缺少键: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate v1 quant config: {e}")
            raise

    def _check_config_consistency(self):
        """
        检查配置一致性，避免用户配置自相矛盾。
        """
        template = self.config['template_config']
        w_bit = self.config.get('w_bit', 8)
        a_bit = self.config.get('a_bit', 8)

        # 检查v1配置中的qconfig是否与w_bit/a_bit一致
        v1_config = template.get('v1', {})
        if v1_config:
            qconfigs = v1_config.get('qconfigs', {})
            for qconfig_name, qconfig in qconfigs.items():
                weight_dtype = qconfig.get('weight', {}).get('dtype', 'int8')
                act_dtype = qconfig.get('act', {}).get('dtype', 'int8')

                # 检查dtype是否与w_bit/a_bit一致
                expected_w_dtype = f'int{w_bit}'
                expected_a_dtype = f'int{a_bit}'

                if weight_dtype != expected_w_dtype:
                    logger.warning(
                        f"Config inconsistency: qconfig '{qconfig_name}' has weight dtype '{weight_dtype}', "
                        f"but w_bit is {w_bit} (expected '{expected_w_dtype}'). "
                        f"Please ensure consistency."
                    )

                if act_dtype != expected_a_dtype:
                    logger.warning(
                        f"Config inconsistency: qconfig '{qconfig_name}' has act dtype '{act_dtype}', "
                        f"but a_bit is {a_bit} (expected '{expected_a_dtype}'). "
                        f"Please ensure consistency."
                    )

    def _generate_quant_config(self, disable_names=None, output_path=None):
        """
        从 config.yaml 中的模板动态生成 modelslim 的配置文件。
        支持 v0 和 v1 两种格式。
        v1是主要格式（AQT场景），v0是legacy格式（非AQT场景）。
        """
        logger.debug("Generating modelslim config file...")

        try:
            template = self.config['template_config']
            
            # 检查配置一致性
            self._check_config_consistency()
            
            # 检查是否有v1配置
            v1_config = template.get('v1')
            if v1_config:
                return self._generate_v1_config(disable_names=disable_names, output_path=output_path)

            # 如果没有v1配置，检查是否有v0配置（legacy格式）
            v0_config = template.get('v0')
            if v0_config:
                return self._generate_v0_config(disable_names=disable_names, output_path=output_path)

            # 兼容旧格式：检查是否有apiversion字段
            api_version = template.get('apiversion', 'modelslim_v0')
            if api_version == 'modelslim_v1':
                return self._generate_v1_config(disable_names=disable_names, output_path=output_path)
            
            # 如果没有明确的v0或v1配置，使用v0格式（兼容旧格式）
            return self._generate_v0_config(disable_names=disable_names, output_path=output_path)

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
