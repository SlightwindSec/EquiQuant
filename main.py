from src.modules import check_requirements
from src.utils import GlobalConfig
from src.core.engine import EquiQuantEngine


if __name__ == "__main__":
    config = GlobalConfig()

    if not check_requirements(config.raw_config["quantization_tool"]):
        exit()

    engine = EquiQuantEngine(config.raw_config)
    engine.run()
