# logger_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path("app.log")

def get_logger(name="resume_app", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # Console handler (useful for terminal)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Rotating file handler (keeps logs limited)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    return logger

# global logger you can import
logger = get_logger()
