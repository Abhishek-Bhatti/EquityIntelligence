"""
Central logging configuration for AlphaQuant.

Call setup_logging() ONCE, at process startup (main.py does this). Every
other module just does:

    import logging
    logger = logging.getLogger(__name__)

and calls logger.debug/info/warning/error/critical(...) — it will
automatically inherit the handlers configured here, since Python's
logging module propagates child loggers ("src.utils.cache_manager") up
to the root logger by default.

Output:
    - Console (stdout): human-readable, same emoji-tagged messages you're
      used to seeing from print().
    - Rotating file (runtimelogs.log, in the project root): same messages,
      plus timestamp + level + logger name, so you can grep/audit later.
      Rotates at 5 MB, keeps 5 backups (runtimelogs.log.1 ... .5), so it
      never grows unbounded.
"""
import logging
import logging.handlers
import os

# Project root = two levels up from this file (src/utils/logging_config.py -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "runtimelogs.log")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int = logging.INFO, log_file_path: str = LOG_FILE_PATH) -> None:
    """
    Configures the root logger with a console handler and a rotating file
    handler. Safe to call more than once — subsequent calls are no-ops,
    so importing a module that calls this at import time won't duplicate
    handlers (which would otherwise print/log everything twice).
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Quiet down noisy third-party libraries so DEBUG-level pipeline logs
    # aren't drowned out by httpx's own request-line logging, etc.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True