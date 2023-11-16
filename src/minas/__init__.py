from pathlib import Path

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent.parent

__all__ = [
    "PACKAGE",
    "PACKAGE_ROOT",
    "director",
]