"""
utils/logger.py
Configures the project-wide Loguru logger.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Hello")
"""

import sys
from pathlib import Path
from loguru import logger


_LOGS_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"


def setup_logger(log_dir: Path = _LOGS_DIR, level: str = "DEBUG") -> None:
    """
    Configure Loguru with:
    - Colored console output
    - Rotating file output (data/logs/app.log)
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler — colored
    logger.add(
        sys.stdout,
        colorize=True,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
    )

    # File handler — rotating, max 10MB, keep 5 files
    logger.add(
        log_dir / "app.log",
        rotation="10 MB",
        retention=5,
        compression="zip",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        encoding="utf-8",
    )

    logger.debug(f"Logger initialized. Log directory: {log_dir}")


def get_logger(name: str):
    """Return a Loguru logger bound to the given module name."""
    return logger.bind(name=name)
