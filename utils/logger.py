"""
Logging utility for hS-IGA-2D simulation.

Usage::

    from utils.logger import logger
    logger.info("message")
    logger.debug("detailed message")
"""

import logging
import os

try:
    from logzero import setup_logger as _setup_logger

    os.makedirs("logs", exist_ok=True)
    logger = _setup_logger(
        name="hS-IGA-logger",
        logfile="logs/simulation.log",
        level=logging.INFO,
        formatter=None,
        fileLoglevel=logging.DEBUG,
        disableStderrLogger=False,
    )
except ImportError:
    # Fallback if logzero is not installed
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("hS-IGA-logger")
    logger.info("logzero not installed — using stdlib logging")
