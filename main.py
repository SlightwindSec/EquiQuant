from src.modules import check_requirements
from src.utils import GlobalConfig
from src.core.engine import EquiQuantEngine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization_tool", required=True, type=str, choices=GlobalConfig.SUPPORTED_QUANTIZATION_TOOLS)
    parser.add_argument("--config_file", required=True, type=str)
    args = parser.parse_args()

    if not check_requirements(args.quantization_tool):
        exit()

    config = GlobalConfig(args.config_file, args.quantization_tool)

    engine = EquiQuantEngine(config.raw_config)
    engine.run()
