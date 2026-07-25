import logging
import os
from logging.handlers import RotatingFileHandler


WD = os.getcwd()
LOG_DIR = os.path.join(WD, "src", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logging(level=logging.INFO, log_file=LOG_FILE):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler_exists = False
    console_handler = None
    normalized_log_file = os.path.abspath(log_file)

    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            if os.path.abspath(handler.baseFilename) == normalized_log_file:
                file_handler_exists = True
                handler.setLevel(level)
                handler.setFormatter(formatter)
        elif isinstance(handler, logging.StreamHandler):
            console_handler = handler

    if not file_handler_exists:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1_000_000,
            backupCount=3,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console_handler is None:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    return logger
