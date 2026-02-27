PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(VENV)/bin/pip
INSTALL_STAMP := $(VENV)/.installed
TEST_FILES := $(shell find tests src -type f \( -name "test_*.py" -o -name "*_test.py" \) 2>/dev/null)

.PHONY: help venv install install-dsvl test run-pipeline run-app clean

help:
	@echo "Targets:"
	@echo "  make venv          Create local virtual environment"
	@echo "  make install       Install project and test tooling"
	@echo "  make install-dsvl  Install optional DeepSeek-VL dependencies"
	@echo "  make test          Run tests (skips cleanly if none exist)"
	@echo "  make run-pipeline  Run full pipeline script"
	@echo "  make run-app       Run Flask app"
	@echo "  make clean         Remove virtual environment and caches"

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV)
	$(PIP_BIN) install --upgrade pip setuptools

venv: $(PYTHON_BIN)

install: $(INSTALL_STAMP)

$(INSTALL_STAMP): $(PYTHON_BIN) pyproject.toml
	$(PIP_BIN) install -e .
	$(PIP_BIN) install pytest
	@touch $(INSTALL_STAMP)

install-dsvl: install
	$(PIP_BIN) install -r requirements-dsvl.txt

test: install
	@if [ -n "$(TEST_FILES)" ]; then \
		$(PYTHON_BIN) -m pytest -q; \
	else \
		echo "No tests found. Skipping pytest."; \
	fi

run-pipeline: install
	$(PYTHON_BIN) run_pipeline.py

run-app: install
	$(PYTHON_BIN) app.py

clean:
	rm -rf $(VENV) .pytest_cache
