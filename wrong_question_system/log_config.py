import json
import logging
import logging.handlers

def get_logger(name,log_file, config_file="config.json"):
    """
    从配置文件中读取日志级别，并创建一个配置好的 logger。
    :param name: logger 名称，通常传入 __name__
    :param config_file: 配置文件路径，默认为 config.json
    :param log_file: 日志文件名称
    :return: 配置好的 logger 对象
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print("加载配置文件出错:", e)
        raise

    # 从配置中获取日志级别，默认为 INFO
    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重复添加 handler（如果 logger 已经配置过）
    if not logger.handlers:
        # 创建 RotatingFileHandler，最大 5MB, 保留 5 个备份
        rf_handler = logging.handlers.RotatingFileHandler(
            log_file, mode="a", encoding="utf-8", maxBytes=5*1024*1024, backupCount=5
        )
        rf_handler.setLevel(log_level)

        # 创建 StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)

        # 定义日志格式，包含模块名和函数名
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s")
        rf_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(rf_handler)
        logger.addHandler(stream_handler)

    logger.info("日志配置成功，日志级别：%s", log_level_str)
    return logger
