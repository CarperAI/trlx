# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "trlX"
copyright = "2023, CarperAI"
author = "CarperAI"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "myst_nb",
    # "myst_parser",
    "sphinx_remove_toctrees",
    "sphinx_copybutton",
    "sphinx_design",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pytorch": ("https://pytorch.readthedocs.io/", None),
}

autodoc_preserve_defaults = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
main_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    # Sometimes sphinx reads its own outputs as inputs!
    "build/html",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

autosummary_generate = True
napolean_use_rtype = False

# -- Options for nbsphinx -----------------------------------------------------

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
# We execute all notebooks, exclude the slow ones using 'exclude_patterns'
nbsphinx_execute = "always"

# Use this kernel instead of the one stored in the notebook metadata:
# nbsphinx_kernel_name = 'python3'

# List of arguments to be passed to the kernel that executes the notebooks:
# nbsphinx_execute_arguments = []

# If True, the build process is continued even if an exception occurs:
# nbsphinx_allow_errors = True


# Controls when a cell will time out (defaults to 30; use -1 for no timeout):
nbsphinx_timeout = 180

# Default Pygments lexer for syntax highlighting in code cells:
# nbsphinx_codecell_lexer = 'ipython3'

# Width of input/output prompts used in CSS:
# nbsphinx_prompt_width = '8ex'

# If window is narrower than this, input/output prompts are on separate lines:
# nbsphinx_responsive_width = '700px'

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r""" # noqa: E501
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. only:: html
    .. role:: raw-html(raw)
        :format: html
    .. nbinfo::
        Interactive online version:
        :raw-html:`<a href="https://colab.research.google.com/github/CarperAI/trlx/blob/master/{{ docname }}"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`
    __ https://github.com/CarperAI/trlx/blob/
        {{ env.config.release }}/{{ docname }}
"""

# This is processed by Jinja2 and inserted after each notebook
# nbsphinx_epilog = r"""
# """

# Input prompt for code cells. "%s" is replaced by the execution count.
# nbsphinx_input_prompt = 'In [%s]:'

# Output prompt for code cells. "%s" is replaced by the execution count.
# nbsphinx_output_prompt = 'Out[%s]:'

# Specify conversion functions for custom notebook formats:
# import jupytext
# nbsphinx_custom_formats = {
#    '.Rmd': lambda s: jupytext.reads(s, '.Rmd'),
# }

# Link or path to require.js, set to empty string to disable
# nbsphinx_requirejs_path = ''

# Options for loading require.js
# nbsphinx_requirejs_options = {'async': 'async'}

# mathjax_config = {
#     'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
# }

# Additional files needed for generating LaTeX/PDF output:
# latex_additional_files = ['references.bib']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "TRLXdoc"

# -- Extension configuration -------------------------------------------------

# Tell sphinx-autodoc-typehints to generate stub parameter annotations including
# types, even if the parameters aren't explicitly documented.
always_document_param_types = True

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # "logo_only": True,
    "show_toc_level": 2,
    "repository_url": "https://github.com/CarperAI/trlx",
    "use_repository_button": True,  # add a "link to repository" button
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/apple-touch-icon-144x144.png"

html_favicon = "_static/favicon-16x16.png"
