import logging
import os

def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("intelligent_tracking_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
