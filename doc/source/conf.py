# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '2D IR Spectroscopy'
copyright = '2023, Julia Brueggemann, GPLv3'
author = 'Julia Brueggemann, C. Jacob, M. Welzel, M. Wolteri, M. Chekmeneva, A. M. van Bodegraven and others'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_mdinclude',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
    ]
source_suffix = ['.rst', '.md'] 
autodoc_member_order = 'bysource'


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = []#['_static']

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
