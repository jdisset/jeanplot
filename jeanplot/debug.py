import logging
import sys
from typing import Optional, Any

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter("%(name)s [%(levelname)s]: %(message)s"))

_root_logger = logging.getLogger("jeanplot")
_root_logger.setLevel(logging.WARNING)  # default level
_root_logger.addHandler(_console_handler)

# global debug state
_debug_enabled = False


def set_debug(enabled: bool = True):
    """enable or disable debug logging globally"""
    global _debug_enabled
    _debug_enabled = enabled
    _root_logger.setLevel(logging.DEBUG if enabled else logging.WARNING)


def get_logger(name: str):
    """get a logger for component/module"""
    return logging.getLogger(f"jeanplot.{name}")


def debug_print(component_id: str, message: str, data: Optional[Any] = None):
    """print debug message with optional data if debug is enabled"""
    if _debug_enabled:
        logger = get_logger(component_id if component_id else "unknown")
        if data is not None:
            logger.debug(f"{message}: {data}")
        else:
            logger.debug(message)
