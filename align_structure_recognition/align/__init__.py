__version__ = '0.9.8'

from .utils import logmanager
logmanager.configure_logging()

__all__ = ["schematic2layout", "CmdlineParser"]


def __getattr__(name):
    if name == "schematic2layout":
        from .main import schematic2layout
        return schematic2layout
    if name == "CmdlineParser":
        from .cmdline import CmdlineParser
        return CmdlineParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
