import logging
import sys
import os


def setup_logger(
    name="EquiQuant",
    log_file=None,
    console_level=logging.INFO,
    file_level=logging.DEBUG,
):
    """
    配置一个日志记录器 (Logger)。

    Args:
        name (str): 日志记录器的名称。
        log_file (str, optional): 日志文件的保存路径。如果为 None，则不记录到文件。
        console_level (int): 控制台输出的最低日志级别。
        file_level (int): 文件输出的最低日志级别。

    Returns:
        logging.Logger: 配置好的日志记录器实例。
    """
    # 1. 创建 logger 实例
    logger = logging.getLogger(name)
    logger.setLevel(
        logging.DEBUG
    )  # 设置 logger 的最低级别为 DEBUG，由 handlers 控制具体输出级别
    logger.propagate = False  # 防止日志向 root logger 传递

    # 避免重复添加 handlers (如果此函数被意外调用多次)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 2. 定义日志格式
    # 示例: 2025-11-14 10:50:00,123 - INFO - quantizer.py - Message
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 3. 配置控制台 Handler (StreamHandler)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 4. 配置文件 Handler (FileHandler)
    if log_file:
        try:
            # 确保日志文件所在的目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # 'a' 模式表示追加 (append)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        except Exception as e:
            logger.error(f"Failed to create log file handler at {log_file}: {e}")

    return logger


DEFAULT_LOG_FILE = "workspace/equiquant_run.log"
# from utils.logger import logger
logger = setup_logger(log_file=DEFAULT_LOG_FILE)
