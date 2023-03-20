# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

logo:
	convert trlx_logo_red.png -define icon:auto-resize=64,48,32,16 _static/favicon.ico
	convert trlx_logo_red.png -resize 16x16 _static/favicon-16x16.png
	convert trlx_logo_red.png -resize 57x57 _static/apple-touch-icon.png
	convert trlx_logo_red.png -resize 57x57 _static/apple-touch-icon-57x57.png
	convert trlx_logo_red.png -resize 72x72 _static/apple-touch-icon-72x72.png
	convert trlx_logo_red.png -resize 76x76 _static/apple-touch-icon-76x76.png
	convert trlx_logo_red.png -resize 114x114 _static/apple-touch-icon-114x114.png
	convert trlx_logo_red.png -resize 120x120 _static/apple-touch-icon-120x120.png
	convert trlx_logo_red.png -resize 144x144 _static/apple-touch-icon-144x144.png
	convert trlx_logo_red.png -resize 152x152 _static/apple-touch-icon-152x152.png
	convert trlx_logo_red.png -resize 180x180 _static/apple-touch-icon-180x180.png

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile logo
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
