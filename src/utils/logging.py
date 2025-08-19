# /amft-project/src/utils/logging.py

import logging
import sys
from typing import Dict, Any


def setup_logging(level=logging.INFO):

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)


def log_metrics(metrics: Dict[str, Any], step: int, prefix: str = "train"):
    logger = logging.getLogger(__name__)
    log_str = f"[{prefix.upper()} Step {step}]"

    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        log_str += f" | {key}: {formatted_value}"

    logger.info(log_str)

