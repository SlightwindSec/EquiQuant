import argparse
from src.utils import GlobalConfig
from src.modules import check_requirements
from src.core.engine import EquiQuantEngine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization_tool", required=True, type=str, choices=GlobalConfig.SUPPORTED_QUANTIZATION_TOOLS)
    parser.add_argument("--config_file", required=True, type=str)
    args = parser.parse_args()

    config = GlobalConfig(args.config_file, args.quantization_tool)
    
    if not check_requirements(args.quantization_tool):
        exit()

    if args.quantization_tool == "modeloptimizer":
        from src.modules.modeloptimizer import ModelOptimizerQuantizer
        modeloptimizer = ModelOptimizerQuantizer(config)
        modeloptimizer.run()
    else:
        engine = EquiQuantEngine(config.raw_config)
        engine.run()
