import setuptools
from pathlib import Path


with open("README.md", "r", encoding='utf-8') as f:
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
        long_description_content="text\markdown",
        url=Path(f'https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}'),
        project_urls={
            "Bug Tracker": Path(f'https"//github.com/{AUTHOR_USER_NAME}/{SRC_REPO}/issues'),
        },
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src")
    )
