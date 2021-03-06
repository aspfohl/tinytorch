PROJECT_NAME=tinytorch
PROJECT_DIRS=tinytorch tests

venv: clean
	python3 -m venv .venv
	poetry run pip install --upgrade pip
	poetry install
	echo "To activate, use 'source ./.venv/bin/activate'"
	echo "To deactivate, use 'deactivate'"

clean: clean_py
	rm -rf .venv
	rm -rf .has-*

clean_py:
	rm -rf ".mypy_cache"
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf .cache .eggs ${PROJECT_NAME}.egg-info dist build

test: test_unit test_format test_style

test_unit:
	poetry run pytest --cov-report term-missing --cov=${PROJECT_NAME}

test_format:
	poetry run black --check ${PROJECT_DIRS}
	poetry run isort --check-only ${PROJECT_DIRS}

test_style:
	poetry run flake8 ${PROJECT_DIRS}
	# poetry run darglint ${PROJECT_DIRS}
	poetry run pylint ${PROJECT_DIRS}
	poetry run mypy --follow-imports=silent --package ${PROJECT_NAME}

format:
	poetry run black ${PROJECT_DIRS}
	poetry run isort ${PROJECT_DIRS}
