import logging

from logging.config import dictConfig
from pathlib import Path
from sys import stdout


APP_NAME = "GIMEval"


def _setup_logfile() -> Path:
    """ensure the logger filepath is in place

    Returns: the logfile Path
    """
    logfile = Path("results/eval.log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.touch(exist_ok=True)
    return logfile


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        },
        "no_datetime": {"format": "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": stdout,
            "formatter": "no_datetime",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": _setup_logfile(),
            "maxBytes": 1024**2 * 10,
            "backupCount": 10,
            "formatter": "standard",
        },
    },
    "loggers": {
        APP_NAME: {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        }
    },
}


def get_logger(name: str | None = None) -> logging.Logger:
    """returns the project logger, scoped to a child name if provided
    Args:
        name: will define a child logger
    """
    dictConfig(LOGGING_CONFIG)

    parent_logger = logging.getLogger(APP_NAME)
    if name:
        return parent_logger.getChild(name)
    return parent_logger
