import yaml

_GLOBAL_CONFIG = None


class GlobalConfig:
    def __init__(self):
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        self.base_model_path = self.raw_config.get("base_model_path")


def get_config():
    global _GLOBAL_CONFIG
    return _GLOBAL_CONFIG


def set_config():
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = GlobalConfig()


def set_global_args():
    set_config()
