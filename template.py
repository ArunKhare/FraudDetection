"""
Initialization Script for Fraud Detection Project.

This script initializes the directory structure and essential configuration files for the 'fraudDetection' project.

Imported Modules:
- logging: Provides a flexible logging infrastructure.
- os: Provides a way to interact with the operating system.
- pathlib.Path: Represents a filesystem path and is used for handling file paths.

Global Variables:
- package_name: Name of the main package for the project.
- list_of_files: List of file paths to be created during project initialization.

Functions:
- setup_logging(): Configures the logging system.
- create_directories_and_files(): Creates necessary directories and empty files.

Usage:
1. Run this script to initialize the project structure and configuration files.

Example:
    python init_setup.py

Note: This script is intended for initializing the project directory structure and should be run once during project setup.
"""

import logging
import os
from pathlib import Path


def setup_logging():
    """Configure logging with INFO level and a specific format."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(message)s:")


package_name = "fraudDetection"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/config/__init__.py",
    f"src/{package_name}/config/configuration.py",
    f"src/{package_name}/pipeline/__init__.py",
    f"src/{package_name}/entity/__init__.py",
    f"src/{package_name}/constants/__init__.py",
    f"src/{package_name}/exception/__init__.py",
    f"src/{package_name}/logger/__init__.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "configs/__init__.py",
    "configs/config.yaml",
    "configs/schema.yaml",
    "configs/model.yaml",
    "dvc.yaml",
    "params.yaml",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "notebook/trials.ipynb",
]


def create_directories_and_files():
    """Create necessary directories and empty files."""
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory : {filedir} for file: {filename}")

        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                pass  # create an empty file
                logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")


if __name__ == "__main__":
    setup_logging()
    create_directories_and_files()
