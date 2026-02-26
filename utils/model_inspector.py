import json
import os
from utils.logger import logger


class ModelInspector:
    @staticmethod
    def get_num_hidden_layers(model_path):
        """
        从 config.json 中读取 'num_hidden_layers'。
        """
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            logger.error(f"config.json not found at {model_path}")
            raise FileNotFoundError(f"config.json not found in {model_path}")
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if "num_hidden_layers" in config_data:
                return config_data["num_hidden_layers"]
            else:
                raise KeyError("'num_hidden_layers' not found in config.json")
        except Exception as e:
            logger.error(f"Failed to parse config.json: {e}")
            raise

    @staticmethod
    def get_layers_by_pattern(model_path, pattern):
        """
        根据模式自动生成层列表。

        Args:
            model_path (str): 基础模型路径 (包含 config.json)。
            pattern (str): 包含 '{i}' 占位符的层名称模式。
                           例如: "model.layers.{i}.mlp.down_proj"

        Returns:
            list: 包含所有层名称的列表。
        """
        logger.info(f"Inspecting model at {model_path} for layer pattern: {pattern}")
        try:
            num_layers = ModelInspector.get_num_hidden_layers(model_path)
            logger.info(f"Found {num_layers} hidden layers.")

            if "{i}" not in pattern:
                logger.warning(
                    f"Layer pattern '{pattern}' does not contain '{{i}}'. Returning as is."
                )
                return [pattern]

            generated_layers = [pattern.format(i=i) for i in range(num_layers)]
            logger.info(
                f"Generated {len(generated_layers)} layer names for fallback strategy."
            )
            return generated_layers

        except Exception as e:
            logger.error(f"Failed to get layers by pattern: {e}")
            return []
