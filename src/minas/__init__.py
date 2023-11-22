import os
from pathlib import Path

from dotenv import load_dotenv

#
PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent.parent

#
# PROJECT_ROOT = os.getcwd()

# # Path to your .env file
# dotenv_path = os.path.join(
#     PROJECT_ROOT,
#     '.env'
# )

# print("Reading .env file from", dotenv_path)

# Load the .env file
# load_dotenv(dotenv_path)

#
__all__ = [
    "PACKAGE",
    "PACKAGE_ROOT",
    "director",
    "constler"
]