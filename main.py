from src.modules import check_requirements
from src.utils import set_global_args, get_config
from src.core.engine import EquiQuantEngine


if __name__ == "__main__":
    if not check_requirements():
        exit()
    set_global_args()
    config = get_config()
    if config.raw_config["quantization_tool"] == "modeloptimizer":
        from src.modules.modeloptimizer import ModelOptimizerQuantizer
        quantizer = ModelOptimizerQuantizer(config)
        quantizer.run()
        exit()
    
    engine = EquiQuantEngine(config.raw_config)
    engine.run()
