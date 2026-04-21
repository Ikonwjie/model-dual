# -*- coding: utf-8 -*-
"""日志记录工具"""

import os
import sys
import logging


def get_logger(log_dir, name, log_filename, level=logging.INFO):
    """
    创建并返回一个日志记录器
    
    Args:
        log_dir: 日志目录
        name: 记录器名称
        log_filename: 日志文件名
        level: 日志级别
    
    Returns:
        logger: 配置好的日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # 文件处理器
    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print("Log directory:", log_dir)

    return logger
