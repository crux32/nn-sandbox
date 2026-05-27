import logging
from typing import Literal

_LOG = logging.getLogger("tensorflow")
_LOG.setLevel(logging.DEBUG)

type LOG_LEVEL = Literal["info", "debug", "error", "warning"]


class Logger:
    @staticmethod
    def log(log_level: LOG_LEVEL, message: str) -> None:
        match log_level:
            case "info":
                _LOG.info(message)
            case "debug":
                _LOG.debug(message)
            case "error":
                _LOG.warning(message)
            case "warning":
                _LOG.warning(message)
