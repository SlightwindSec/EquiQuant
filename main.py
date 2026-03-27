from .modules import check_requirements
from .utils import set_global_args, get_config
from .core.engine import EquiQuantEngine


if __name__ == "__main__":
    if not check_requirements():
        exit()
    set_global_args()
    config = get_config()
    engine = EquiQuantEngine(config.raw_config)
    engine.run()
