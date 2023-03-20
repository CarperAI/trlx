#!/bin/bash

`which sphinx-build` -T -E -b html -d _build/doctrees-readthedocs -D language=en . _build/html
