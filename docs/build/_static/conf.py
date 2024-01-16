# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(1, os.path.abspath("../src"))

# Get the absolute path to the root directory (assuming conf.py is in Root/docs/source)
# join_path = lambda *paths: os.path.join(os.path.dirname(__file__),*paths)

_static_dir = os.path.abspath(os.path.dirname(__file__)) or "."
_source_dir = os.path.join(_static_dir, "..")
docs_dir = os.path.join(_source_dir, "..")
root_dir = os.path.join(docs_dir, "..")

src_dir = os.path.join(root_dir, "src")

# root_doc = os.path.join(_static_dir, "index")
# # Add the root directory to sys.path
sys.path.insert(0, root_dir)
sys.path.insert(1, src_dir)
sys.path.insert(2, docs_dir)
sys.path.insert(3, _static_dir)

# I hope this in correct in conf.py settings
# _static contains conf.py in index.rst

# My package namespace packages are in root_dir and src_dir

# src has no __init__py
# fraudDetection contains __init__py and modules and submodules have __init__.py in each folder

# The commands that i have used so far are:
# sphinx-apidoc -o  docs/source .
# sphinx-apidoc -o  docs/ ./src
# sphinx-build -M  html docs/source  docs/build
# could build the source rst  succefully with a single command using:
# sphinx-apidoc -M html --implicit-namespaces -o docs/source (reason : error in importing packages as each package was added with pre-fix of MY root project folder FraudDetection which is root_dir here)
# name would be like FraudDetection/src/fraudDetection/<modules> where as my imports are from fraudetection.<modules>
# The root contains:
#     - apps / __init__.py, trainingapp.py, app.py
#     - tests / __init_py, unit/test.py, integration/test.py
#     - configs/ config.ymal, model.yaml etc..

# The src contains:
#     - fraudDetection/ __init__.py, prediction_service/ __init__.py, prediction.py
#                                    training/ __init__py, dataingestion.py, datavalidation.py and so on

project = "Fraud-Detection"
copyright = "2024, Arun khare"
author = "Arun khare"
release = "beta:v1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
exclude_patterns = [
    "*flake.rst",
    "configs.rst",
    "setup.rst",
    "*.md",
]

html_theme = "sphinx_rtd_theme"
html_static_path = [_static_dir]
