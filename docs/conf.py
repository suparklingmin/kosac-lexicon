"""Sphinx configuration for the kosac-lexicon documentation."""
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Make the package importable for autodoc even without an install.
sys.path.insert(0, os.path.abspath("../src"))

project = "kosac-lexicon"
author = "Sumin Park"
copyright = "2016–2026, Sumin Park"

try:
    release = _pkg_version("kosac-lexicon")
except PackageNotFoundError:
    release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Optional/heavy dependencies that need not be installed to build the docs.
autodoc_mock_imports = ["kiwipiepy", "transformers", "sklearn"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"kosac-lexicon {release}"
