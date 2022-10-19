IGNORE_PEP=E203,E221,E241,E272,E501,F811

.PHONY:  style quality

check_dirs := trlx/

style:
	black $(check_dirs)
	isort $(check_dirs) # see pyproject.toml for isort config
	flake8 $(check_dirs) --ignore=$(IGNORE_PEP)

quality:
	isort --check-only $(check_dirs) # see pyproject.toml for isort config
	flake8 $(check_dirs)
