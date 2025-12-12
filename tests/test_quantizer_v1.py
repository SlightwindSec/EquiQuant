import os
import sys
import tempfile
import unittest

try:
    from ruamel.yaml import YAML
except ImportError:  # pragma: no cover - handled by skip condition
    YAML = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.quantizer import ModelslimQuantizer, RUAMEL_AVAILABLE


@unittest.skipUnless(RUAMEL_AVAILABLE and YAML is not None, "ruamel.yaml is required for modelslim_v1 formatting")
class TestModelslimV1Config(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_quant_config(self):
        return {
            "template_config": {
                "w_bit": 8,
                "a_bit": 8,
                "v1": {
                    "metadata": {
                        "config_id": "qwen3-dense-w8a8-v1",
                        "score": 90,
                        "label": {"is_sparse": False, "kv_cache": False},
                    },
                    "qconfigs": {
                        "default_w8a8_dynamic": {
                            "act": {
                                "scope": "per_token",
                                "dtype": "int8",
                                "symmetric": True,
                                "method": "minmax",
                            },
                            "weight": {
                                "scope": "per_channel",
                                "dtype": "int8",
                                "symmetric": True,
                                "method": "minmax",
                            },
                        },
                        "default_w4a8_dynamic": {
                            "act": {
                                "scope": "per_token",
                                "dtype": "int8",
                                "symmetric": True,
                                "method": "minmax",
                            },
                            "weight": {
                                "scope": "per_group",
                                "dtype": "int4",
                                "symmetric": True,
                                "method": "minmax",
                                "ext": {"group_size": 64},
                            },
                        },
                    },
                    "process": [
                        {
                            "type": "group",
                            "configs": [
                                {
                                    "type": "linear_quant",
                                    "qconfig_ref": "default_w4a8_dynamic",
                                    "include": [],
                                    "exclude": [],
                                },
                                {
                                    "type": "linear_quant",
                                    "qconfig_ref": "default_w8a8_dynamic",
                                    "include": [],
                                    "exclude": [],
                                },
                            ],
                        }
                    ],
                    "save": [{"type": "ascendv1_saver", "part_file_size": 4}],
                },
            },
            "w_bit": 8,
            "a_bit": 8,
            "model_type": "Qwen3-32B",
            "visible_devices": "0",
            "device": "npu",
            "trust_remote_code": True,
        }

    def test_v1_yaml_outputs_anchors_and_aliases(self):
        quant_config = self._build_quant_config()
        output_path = os.path.join(self.tmpdir.name, "modelslim_v1.yaml")

        quantizer = ModelslimQuantizer(
            quant_config=quant_config,
            base_model_path="/tmp/dummy-model",
            fallback_layers=[],
            output_config_path=output_path,
            output_weights_path=os.path.join(self.tmpdir.name, "quantized"),
        )

        generated_path = quantizer.generate_config_only(output_path=output_path)
        self.assertTrue(os.path.exists(generated_path))

        with open(generated_path, "r", encoding="utf-8") as f:
            yaml_text = f.read()

        # Anchors should be emitted (not flattened) and aliases should not be quoted strings.
        self.assertIn("default_w8a8_dynamic: &default_w8a8_dynamic", yaml_text)
        self.assertIn("default_w4a8_dynamic: &default_w4a8_dynamic", yaml_text)
        self.assertNotIn("qconfig: '*default_w4a8_dynamic'", yaml_text)
        self.assertNotIn("qconfig: \"*default_w4a8_dynamic\"", yaml_text)

        # Parsed YAML should preserve alias references instead of duplicating mappings.
        yaml = YAML()
        doc = yaml.load(yaml_text)
        process_cfgs = doc["spec"]["process"][0]["configs"]
        self.assertIs(process_cfgs[0]["qconfig"], doc["default_w4a8_dynamic"])
        self.assertIs(process_cfgs[1]["qconfig"], doc["default_w8a8_dynamic"])


if __name__ == "__main__":
    unittest.main()
