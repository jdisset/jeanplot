import logging
import sys

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter("JEANPLOT [%(name)s] [%(levelname)s]: %(message)s"))

_root_logger = logging.getLogger("jeanplot")
_root_logger.propagate = False
_root_logger.setLevel(logging.WARNING)
if _root_logger.hasHandlers():
    _root_logger.handlers.clear()
_root_logger.addHandler(_console_handler)

_debug_enabled = False


def set_debug(enabled: bool = True):
    """enable or disable debug logging globally."""
    global _debug_enabled
    _debug_enabled = enabled
    level = logging.DEBUG if enabled else logging.INFO
    _root_logger.setLevel(level)
    for handler in _root_logger.handlers:
        handler.setLevel(level)


def get_logger(name: str):
    return logging.getLogger(f"jeanplot.{name}")


def debug_print(source_id: str, message: str, data=None):
    import numpy as np

    logger = get_logger(source_id if source_id else "unknown")
    if logger.isEnabledFor(logging.DEBUG):
        if data is not None:
            if isinstance(data, np.ndarray):
                data_str = np.array2string(data, precision=3, suppress_small=True, prefix="  ")
                logger.debug(f"{message}\nData:\n{data_str}")
            else:
                logger.debug(f"{message}: {data}")
        else:
            logger.debug(message)


class DebugMixin:
    def _log_debug(self, message: str, data=None):
        debug_print(getattr(self, "id", None) or self.__class__.__name__, message, data)
