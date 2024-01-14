# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(1, os.path.abspath("../src"))

# Get the absolute path to the root directory (assuming conf.py is in Root/docs/source)
root_dir = os.path.abspath(
    os.path.join((os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))), "..")
)
src_dir = os.path.join(root_dir, "src")
# # Add the root directory to sys.path
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.abspath(src_dir))

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
html_static_path = ["_static"]
