import os
from pathlib import Path

from dotenv import load_dotenv

# Path to your .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# Load the .env file
load_dotenv(dotenv_path)

#
PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent.parent

__all__ = [
    "PACKAGE",
    "PACKAGE_ROOT",
    "director",
    "constler"
]