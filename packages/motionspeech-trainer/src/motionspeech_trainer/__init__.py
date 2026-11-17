# Modified for DuoFlow; see NOTICE.md.
import logging
import os
import sys
from logging import getLogger
from pathlib import Path

from rich.logging import RichHandler

# Get the process rank
IS_MULTI_GPU = os.environ.get("LOCAL_RANK") is not None
RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Configure with Rich
logging.basicConfig(
    level="INFO",
    format=f"\\[rank {RANK}] %(message)s" if IS_MULTI_GPU else "%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
    ],
)

# Get the logger and configure it
logger = getLogger("motionspeech_trainer")
logger.setLevel(logging.INFO)
logger.propagate = True

# Set level based on process
if RANK != 0:
    logger.setLevel(logging.WARNING)

# Expose common logging functions directly
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical


def add_file_handler(log_path: str | Path) -> None:
    """Add a file handler to the root logger if one doesn't already exist."""
    root_logger = logging.getLogger()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return

    format_string = f"[rank {RANK}] %(message)s" if IS_MULTI_GPU else "%(message)s"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(file_handler)


# Add the root directory to the Python path so we can import from scripts.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
