"""Process signal and logging setup for the Agent entrypoint."""

import gzip
import logging
import os
import shutil
import signal
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from src.retrieval.config import (
    LOG_DIR,
    LOG_LEVEL,
    LOG_RETENTION_DAYS,
    PROJECT_ROOT,
)


def _force_exit(*_: object) -> None:
    # gRPC background threads can keep the worker alive after normal signal handling.
    os._exit(1)


def _compressed_name(default_name: str) -> str:
    return default_name + ".gz"


def _compress_rotated_log(source: str, destination: str) -> None:
    with (
        open(source, "rb") as source_file,
        gzip.open(destination, "wb") as destination_file,
    ):
        shutil.copyfileobj(source_file, destination_file)
    os.remove(source)


def configure_process_runtime() -> None:
    """Install process signals and the shared console/file logging handlers."""
    try:
        signal.signal(signal.SIGINT, _force_exit)
        signal.signal(signal.SIGTERM, _force_exit)
    except ValueError:
        pass

    log_dir = Path(LOG_DIR) if Path(LOG_DIR).is_absolute() else PROJECT_ROOT / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    file_handler = TimedRotatingFileHandler(
        filename=str(log_dir / "app.log"),
        when="midnight",
        interval=1,
        backupCount=LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    file_handler.namer = _compressed_name
    file_handler.rotator = _compress_rotated_log
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        handlers=[console_handler, file_handler],
    )
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
