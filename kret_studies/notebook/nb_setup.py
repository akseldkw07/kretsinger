import logging


def get_notebook_logger(name: str = "kret_nb", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger configured for notebook usage.
    - Logs to stdout with a simple format.
    - Avoids duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging to root
    # Remove all existing handlers to avoid duplicates
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
