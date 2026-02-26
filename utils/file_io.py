import yaml
import os
from utils.logger import logger


def read_yaml(file_path):
    """
    读取 YAML 文件并返回 Python 字典。

    Args:
        file_path (str): YAML 文件的路径。

    Returns:
        dict: 解析后的数据。

    Raises:
        FileNotFoundError: 如果文件未找到。
        yaml.YAMLError: 如果文件解析失败。
    """
    if not os.path.exists(file_path):
        logger.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    logger.debug(f"Reading YAML file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # 使用 SafeLoader 防止执行任意代码
            data = yaml.safe_load(f)
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
        raise


def write_yaml(data, file_path):
    """
    将 Python 字典写入 YAML 文件。

    Args:
        data (dict): 要写入的数据。
        file_path (str): 目标 YAML 文件的路径。

    Returns:
        bool: True 表示成功, False 表示失败。
    """
    logger.debug(f"Writing YAML file: {file_path}")
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

        with open(file_path, "w", encoding="utf-8") as f:
            # Dumper=yaml.SafeDumper 可以确保输出是标准的 YAML 格式
            # sort_keys=False 保持字典中的插入顺序，对 modelslim 配置更友好
            yaml.dump(
                data, f, Dumper=yaml.SafeDumper, sort_keys=False, allow_unicode=True
            )

        logger.info(f"Successfully generated config file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write YAML file {file_path}: {e}")
        return False


def ensure_dir_exists(dir_path):
    """
    确保指定的目录存在，如果不存在则创建它。
    """
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
        except Exception as e:
            logger.error(f"Could not create directory {dir_path}: {e}")
            raise


def clean_directory(dir_path):
    """
    清空指定目录下的所有内容 (删除并重建)。
    """
    import shutil

    logger.warning(f"Cleaning directory: {dir_path}")
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to clean directory {dir_path}: {e}")
        return False
