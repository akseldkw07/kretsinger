import logging
import os
import sys

from dotenv import load_dotenv


def find_nearest_dotenv(start_dir: str = "."):
    """
    Finds the nearest .env file starting from the given directory and going up the directory tree.

    Args:
        start_dir (str): The directory to start searching from. Defaults to the current directory.

    Returns:
        str | None: The path to the nearest .env file, or None if not found.
    """
    current_dir = os.path.abspath(start_dir)
    while current_dir != os.path.dirname(current_dir):  # Stop when we reach the root
        dotenv_path = os.path.join(current_dir, ".env")
        if os.path.isfile(dotenv_path):
            return dotenv_path
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("No .env file found")


def load_dotenv_file(dotenv_path: str | None = None):
    """
    Loads environment variables from a .env file.
    """
    if dotenv_path is None:
        dotenv_path = find_nearest_dotenv()
    if not os.path.isfile(dotenv_path):
        raise FileNotFoundError(f"No .env file found at {dotenv_path}")

    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}.")


def get_notebook_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s | %(levelname)s | %(name)s ] %(message)s",
        stream=sys.stdout,
        force=True,  # <- crucial in notebooks; overrides prior handlers
    )


# def get_notebook_logger(level: int = logging.INFO) -> logging.Logger:
#     """
#     Sets up logging so that:
#     - All INFO and above go to the specified log file (from NB_LOGFILE env or 'notebook.log').
#     - All WARNING and above also go to stdout (notebook output).
#     - All loggers default to INFO.
#     Returns a logger for notebook usage (root logger).
#     """
#     logfile = os.environ.get("NB_LOGFILE", "notebook.log")
#     print(logfile)
#     root_logger = logging.getLogger()
#     root_logger.setLevel(level)

#     # Remove all handlers to avoid duplicates
#     while root_logger.handlers:
#         root_logger.removeHandler(root_logger.handlers[0])

#     # File handler for all INFO and above
#     fh = logging.FileHandler(logfile, mode="a")
#     fh.setLevel(logging.INFO)
#     fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
#     root_logger.addHandler(fh)

#     # Stream handler for WARNING and above
#     sh = logging.StreamHandler()
#     sh.setLevel(logging.WARNING)
#     sh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
#     root_logger.addHandler(sh)

#     # Optional: log that logging is set up
#     root_logger.info(f"Notebook logging initialized. Log file: {os.path.abspath(logfile)}")
#     return root_logger
