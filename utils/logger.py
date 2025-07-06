'''
Author: Zheng Wang zwang3478@gatech.edu
Date: 2023-09-20 10:10:14
LastEditors: Zheng Wang zwang3478@gatech.edu
LastEditTime: 2023-09-20 10:10:38
FilePath: /QPLoRA/utils/logger.py
'''

import logging
import os

# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name, log_dir):
    # Remove log dir if it exists, or create if not
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)  # Clears old logs
    os.makedirs(log_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
