"""
fraudDetection: A Python project for online Banking transaction fraud detection.

This module contains the setup script for packaging and distributing the 'fraudDetection' project.

Functions:
- setup(): Configures the project metadata and dependencies for packaging.

Usage:
1. Run this script to configure the project metadata.
2. Use 'setuptools' to package and distribute the project.

Example:
    from setuptools import setup

    # (Your project metadata and dependencies here)

    setup(
        name="fraudDetection",
        version="0.0.0",
        author="ArunKhare",
        description="A python project for online Banking transaction fraud detection",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ArunKhare/fraudDetection",
        project_urls={
            "Bug Tracker": "https://github.com/ArunKhare/fraudDetection/issues",
        },
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
    )
"""

import setuptools
from pathlib import Path


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

    __version__ = "0.0.0"

    REPO_NAME = "FraudDectection"
    AUTHOR_USER_NAME = "ArunKhare"
    SRC_REPO = "fraudDetection"
    AUTHOR_EMAIL = "arunvkhare@gmail.com"

    setuptools.setup(
        name=SRC_REPO,
        version=__version__,
        author=AUTHOR_USER_NAME,
        description="A python project for online Banking transaction fraud detection",
        long_description=long_description,
        long_description_content="text markdown",
        url=Path(f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}"),
        project_urls={
            "Bug Tracker": Path(
                f'https"//github.com/{AUTHOR_USER_NAME}/{SRC_REPO}/issues'
            ),
        },
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
    )
