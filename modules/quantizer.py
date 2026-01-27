import os
import abc
import copy
import json
import yaml
from utils.shell import ShellRunner
from utils.file_io import write_yaml
from utils.logger import logger
from collections import defaultdict
from aqt.utils.quant_config_manager import TRANSFORMER_LAYER_PATTERNS

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    RUAMEL_AVAILABLE = True
except ImportError:
    RUAMEL_AVAILABLE = False
    logger.warning("ruamel.yaml not available. v1 format support may be limited.")


class FlowStyleList(list):
    pass


def flow_style_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowStyleList, flow_style_representer)


class BaseQuantizer(abc.ABC):
    """
    封装 量化器。
    """
    def __init__(
        self,
        quant_config: dict,
        base_model_path: str,
        fallback_layers: list[str],
        output_config_path: str,
        output_weights_path: str,
        hybrid_quant_schema_path: str,
    ):
        """
        Args:
            quant_config (dict): 来自 config.yaml['quantization']
            base_model_path (str): 来自 config.yaml['base_model_path']
            fallback_layers (list): 本次回退层列表
            output_config_path (str): 本次运行 modelslim YAML 或 llmcompressor PY脚本的保存路径
            output_weights_path (str): 本次运行量化权重输出路径
            hybrid_quant_schema_path (str): 本次aqt得到的混合量化配置路径
        """
        self.config = quant_config
        self.base_model_path = base_model_path
        self.fallback_layers = fallback_layers
        self.output_config_path = output_config_path
        self.output_weights_path = output_weights_path
        self.hybrid_quant_schema_path = hybrid_quant_schema_path

        self._generate_quant_config()

    @abc.abstractclassmethod
    def _generate_quant_config(self):
        pass

    def run(self):
        pass


class ModelslimQuantizer(BaseQuantizer):
    """
    封装 Modelslim 量化工具。
    """

    def _generate_base_v1_config(self, disable_names=None):
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

            with open(self.output_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(root, f)

            logger.info(f"Successfully generated v1 config file: {self.output_config_path}")

        except KeyError as e:
            logger.error(f"Config Error: 'config.yaml' 中 'quantization.template_config' 缺少键: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate v1 quant config: {e}")
            raise
    
    # FIXME: modelslim yaml 填充bug
    def _fill_modelslim_yaml(self) -> None:
        with open(self.output_config_path) as f:
            base = yaml.safe_load(f)      
        with open(self.hybrid_quant_schema_path) as f:
            hybrid_quant_schema = json.load(f)
        w4a8_cfg = defaultdict(FlowStyleList)
        w8a8_cfg = defaultdict(FlowStyleList)

        for pattern, quant_schema in hybrid_quant_schema.items():
            if pattern in TRANSFORMER_LAYER_PATTERNS:
                if quant_schema.startswith("w4"):
                    w4a8_cfg["include"].append(pattern)
                elif quant_schema.startswith("w8"):
                    w8a8_cfg["include"].append(pattern)
            else:
                if quant_schema.startswith("w4"):
                    w4a8_cfg["include"].append(pattern)
                    if "layers.*.mlp.experts" not in pattern:
                        w8a8_cfg["exclude"].append(pattern)
                elif quant_schema.startswith("w8"):
                    if "layers.*.mlp.experts" not in pattern:
                        w4a8_cfg["exclude"].append(pattern)
                    w8a8_cfg["include"].append(pattern)
                else:
                    w8a8_cfg["exclude"].append(pattern)

        anchor_mapping = {}
        for process in base["spec"]["process"]:
            if process["type"] == "group":
                for config in process["configs"]:
                    if config["qconfig"]["weight"]["dtype"] == "int4":
                        config.update(w4a8_cfg)
                        anchor_mapping["default_w8a8_dynamic"] = id(config)
                    elif config["qconfig"]["weight"]["dtype"] == "int8":
                        config.update(w8a8_cfg)
                        anchor_mapping["default_w4a8_dynamic"] = id(config)

        with open(self.output_config_path, "w") as f:
            yaml.dump(base, f, default_flow_style=False, sort_keys=False)
        

    def _generate_quant_config(self, disable_names=None):
        """
        从 config.yaml 中的模板动态生成 modelslim 的配置文件。
        支持 v0 和 v1 两种格式。
        v1是主要格式（AQT场景），v0是legacy格式（非AQT场景）。
        """
        logger.debug("Generating modelslim config file...")

        try:
            self._generate_base_v1_config(disable_names=disable_names)
            self._fill_modelslim_yaml()
        except KeyError as e:
            logger.error(f"Config Error: 'config.yaml' 中 'quantization.template_config' 缺少键: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate quant config: {e}")
            raise

    def run(self):
        """
        执行完整的量化流程。
        """
        logger.info(f"Starting quantization with msmodelslim... Fallback layers: {len(self.fallback_layers)}")
        try:
            env_prefix = f"export ASCEND_RT_VISIBLE_DEVICES={self.config['visible_devices']}; "
            cmd = (
                f"msmodelslim quant "
                f"--model_path {self.base_model_path} " 
                f"--save_path {self.output_weights_path} "
                f"--config_path {self.output_config_path} "
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


class LLMCompressorQuantizer(BaseQuantizer):
    '''
    封装 LLMCompressor 量化工具。
    '''

    def _generate_llmcompressor_config(self) -> None:
        with open(self.hybrid_quant_schema_path) as f:
            hybrid_quant_schema = json.load(f)
        targets = {
            'w8a8_dynamic': [],
            'w4a8_dynamic': [],
        }
        for pattern, quant_schema in hybrid_quant_schema.items():
                if pattern in TRANSFORMER_LAYER_PATTERNS:
                    if quant_schema.startswith("w4"):
                        targets["w4a8_dynamic"].append(pattern)
                    elif quant_schema.startswith("w8"):
                        targets["w8a8_dynamic"].append(pattern)
                else:
                    if quant_schema.startswith("w4"):
                        targets["w4a8_dynamic"].append(pattern)
                    elif quant_schema.startswith("w8"):
                        targets["w8a8_dynamic"].append(pattern)
                    else:
                        targets["w8a8_dynamic"].append(pattern)
        # FIXME: quant_schema 添加进脚本中 / 校准集加载逻辑修改
        quant_schema = {
            "w8a8_dynamic": {
                "targets": targets["w8a8_dynamic"],
                "weights": {
                    "num_bits": 8,
                    "type": "QuantizationType.INT",
                    "strategy": "QuantizationStrategy.CHANNEL",
                    "symmetric": True,
                    "dynamic": False
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "QuantizationType.INT",
                    "strategy": "QuantizationStrategy.TOKEN",
                    "symmetric": True,
                    "dynamic": True
                },
            },
            "w4a8_dynamic": {
                "targets": targets["w4a8_dynamic"],
                "weights": {
                    "num_bits": 4,
                    "type": "QuantizationType.INT",
                    "strategy": "QuantizationStrategy.GROUP",
                    "symmetric": True,
                    "dynamic": False,
                    "group_size": 64
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "QuantizationType.INT",
                    "strategy": "QuantizationStrategy.TOKEN",
                    "symmetric": True,
                    "dynamic": True
                },
            },
        }

        script_lines = []

        script_lines.extend([
            "import os",
            ""
            "import torch",
            "from datasets import load_dataset",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            "from llmcompressor import oneshot",
            "from llmcompressor.modifiers.awq import AWQModifier"
            "from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier"
            "from llmcompressor.modifiers.smoothquant import SmoothQuantModifier",
            "from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationType, QuantizationStrategy",
            "",
            "",
        ])
        
        script_lines.extend([
            f"MODEL_ID = {repr(self.base_model_path)}",
            f"model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=\"auto\")",
            f"tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)",
            "",
        ])
        
        need_calib_data = self.config['modifier'] != 'PTQ' or self.config['enable_smoothquant']

        modifier_map = {
            'AWQ': 'AWQModifier',
            'PTQ': 'QuantizationModifier',
            'GPTQ': 'GPTQModifier',
        }

        if need_calib_data:
            script_lines.extend([
                f"DATASET_ID = {repr(self.config['calib_dataset_path'])}",
                f"DATASET_SPLIT = {repr('train_sft')}",
                "",
                f"NUM_CALIBRATION_SAMPLES = {self.config['num_calibration_samples']}",
                f"MAX_SEQUENCE_LENGTH = {self.config['max_sequence_length']}",
                ""
                f"ds = load_dataset(DATASET_ID, split=f\"{{DATASET_SPLIT}}[:{{NUM_CALIBRATION_SAMPLES}}]\")",
                f"ds = ds.shuffle(seed=42)",
                "",
                "",
                "def preprocess(example):",
                "    return {",
                "        \"text\": tokenizer.apply_chat_template(",
                "            example[\"messages\"],",
                "            tokenize=False,",
                "        )",
                "    }",
                "",
                "",
                "ds = ds.map(preprocess)",
                "",
                "",
                "def tokenize(sample):",
                "    return tokenizer(",
                "        sample[\"text\"],",
                "        padding=False,",
                "        max_length=MAX_SEQUENCE_LENGTH,",
                "        truncation=True,",
                "        add_special_tokens=False,",
                "    )",
                "",
                "",
                "ds = ds.map(tokenize, remove_columns=ds.column_names)",
                "",
                "",
            ])

        script_lines.extend([
            "ignore = ['lm_head', 're:.*mlp.gate$', 'model.embed_tokens']",
            "recipe = [",
        ])

        if self.config['enable_smoothquant']:
            script_lines.append(f"    SmoothQuantModifier(smoothing_strength={self.config['smoothing_strength']}),")
        script_lines.extend([
            f"    {modifier_map[self.config['modifier']]}(",
            "        config_groups=quant_schema,",
            "        ignore=ignore,",
            "    ),",
            "]",
            "",
            "",
        ])
        
        if need_calib_data:
            script_lines.extend([
                "oneshot(",
                "    model=model,",
                "    dataset=ds,",
                "    recipe=recipe,",
                "    tokenizer=tokenizer,",
                "    max_seq_length=MAX_SEQUENCE_LENGTH,",
                "    num_calibration_samples=NUM_CALIBRATION_SAMPLES,",
                f"    output_dir={repr(self.output_weights_path)}"
                ")",
                "",
                "",
            ])
        else:
            script_lines.extend([
                "oneshot(",
                "    model=model,",
                "    recipe=recipe,",
                "    tokenizer=tokenizer,",
                f"    output_dir={repr(self.output_weights_path)}"
                ")",
                "",
                "",
            ])
        
        script_lines.extend([
            f"SAVE_DIR = {repr(self.output_weights_path)}",
            "model.save_pretrained(SAVE_DIR, save_compressed=True)",
            "tokenizer.save_pretrained(SAVE_DIR)",
        ])

        final_script = "\n".join(script_lines)
        

        with open(self.output_config_path, "w", encoding="utf-8") as f:
            f.write(final_script)

    def _generate_quant_config(self, disable_names=None):
            logger.debug("Generating llmcompressor config file...")
            try:
                self._generate_llmcompressor_config()
                logger.info(f"scucceded to generate quant config: {self.output_config_path}")
            except Exception as e:
                logger.error(f"Failed to generate quant config: {e}")
                raise

    def run(self):
        """
        执行完整的量化流程。
        """
        logger.info(f"Starting quantization with llmcompressor... Fallback layers: {len(self.fallback_layers)}")
        try:
            env_prefix = f"export ASCEND_RT_VISIBLE_DEVICES={self.config['visible_devices']}; "
            cmd = (
                f"python {self.output_config_path} "
            )
            full_cmd = env_prefix + cmd
            success, stdout, stderr = ShellRunner.run_cmd(full_cmd, timeout=10800)
            if not success:
                logger.error(f"LLMCompressor quantization failed. Stderr: {stderr}")
                raise Exception(f"LLMCompressor failed.")
            logger.info("Quantization finished successfully.")
            return self.output_weights_path
        except Exception as e:
            logger.error(f"An error occurred during quantization run: {e}")
            return None
