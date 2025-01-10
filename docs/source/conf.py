# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SelfiSys"
copyright = "2025, Tristan Hoellinger"
author = "Tristan Hoellinger"
release = (
    "/Users/hoellinger/Library/CloudStorage/Dropbox/travail/these/science/code/SELFI/selfisys/src/"
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings.
    "sphinx.ext.viewcode",  # Adds links to highlighted source code.
    "sphinx.ext.napoleon",  # Supports Google and NumPy-style docstrings.
    "myst_parser",  # Markdown support.
    "sphinx.ext.autosummary",  # Generate summary tables for API docs.
    "sphinx.ext.mathjax",  # For rendering mathematical equations using MathJax.
    "sphinx.ext.todo",  # Support for TODO notes.
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = ["_static"]

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "style_external_links": True,
    "titles_only": False,
}

# Add GitHub link
html_context = {
    "display_github": True,  # Enables the link
    "github_user": "hoellin",  # GitHub username
    "github_repo": "selfisys_public",  # Repository name
    "github_version": "main",  # Branch name
    "conf_py_path": "/docs/source/",  # Path to the documentation directory in the repo
}

import os
import sys
import shutil

# Add the project's source path to sys.path
sys.path.insert(0, os.path.abspath("../../src"))

# Copy Markdown files from parent directory
markdown_files = ["README.md", "CONTRIBUTING.md", "REFERENCES.md"]
for md_file in markdown_files:
    src = os.path.abspath(f"../../{md_file}")  # Adjust the path if necessary
    dst = os.path.join(os.path.dirname(__file__), os.path.basename(md_file))
    try:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
    except Exception as e:
        print(f"Error copying {md_file}: {e}")


def generate_rst_files():
    """
    Automatically generate .rst files for standalone Python modules.
    Avoids regeneration if files already exist.
    """
    source_dir = os.path.abspath("src/selfisys")  # Adjust to your source directory
    output_dir = os.path.join(os.path.dirname(__file__))  # docs/source directory

    # Ensure source_dir exists
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"The source directory '{source_dir}' does not exist.")

    # List all Python files in the source directory (excluding __init__.py)
    for filename in os.listdir(source_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]  # Remove .py extension
            output_file = os.path.join(output_dir, f"selfisys.{module_name}.rst")

            # Skip regeneration if the file already exists
            if os.path.exists(output_file):
                continue

            # Write the .rst file content
            with open(output_file, "w") as f:
                # Module header
                f.write(f"{module_name}\n")
                f.write("=" * len(module_name) + "\n\n")
                f.write(f".. automodule:: selfisys.{module_name}\n")
                f.write("   :members:\n")
                f.write("   :undoc-members:\n")
                f.write("   :show-inheritance:\n")

            print(f"Generated {output_file}")


# Register a Sphinx hook to generate .rst files before the build
def setup(app):
    app.connect("builder-inited", lambda app: generate_rst_files())
