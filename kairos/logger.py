import copy
import logging
from pathlib import Path
import uvicorn

BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"
GREY = "\033[90m"


class KairosFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: "",
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        process_name = self.get_process_name(record.pathname)
        record.prefix = f"{CYAN}({process_name} pid={record.process}){RESET}"

        color = self.LEVEL_COLORS.get(record.levelno, "")

        record.levelname = f"{color}{record.levelname}{RESET}"
        record.location = (
            f"{GREY}{self.formatTime(record, self.datefmt)} "
            f"[{record.filename}:{record.lineno}]{RESET}"
        )

        return super().format(record)

    def get_process_name(self, pathname: str) -> str:
        path_elements = Path(pathname).parts
        if "core" in path_elements or ("ipc" in path_elements and "server" in path_elements):
            return "KairosCore"
        else:
            return "KairosAPI"


def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(
        KairosFormatter(
            "%(prefix)s %(levelname)s: %(location)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def build_uvicorn_log_config() -> dict:
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)

    log_config["formatters"]["default"]["fmt"] = (
        f"{CYAN}(KairosAPI pid=%(process)d){RESET} %(levelprefix)s %(message)s"
    )

    # optional, but keeps Uvicorn's colored INFO/WARNING/ERROR prefixes
    log_config["formatters"]["default"]["use_colors"] = True

    return log_config

