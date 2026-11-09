UV ?= uv
FLAGS ?=
MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
VENV ?= .venv
SDK_ZIP ?= llm_sdk.zip

.PHONY: help install sdk-install check-env run debug clean test lint lint-strict

help:
	@echo "Available targets:"
	@echo "  make install       Recreate $(VENV) from scratch (CPU torch, from uv.lock)"
	@echo "  make sdk-install   Unpack llm_sdk from llm_sdk.zip (or SDK_ZIP=<zip>)"
	@echo "  make run           Run the project"
	@echo "  make debug         Run the project with pdb"
	@echo "  make clean         Remove caches"
	@echo "  make test          Run the test program"
	@echo "  make lint          Run flake8 and required mypy checks"
	@echo "  make lint-strict   Run flake8 and mypy --strict"

# --reinstall-package llm-sdk: uv otherwise reuses a stale cached build of the local SDK.
install:
	@set -eu; \
	rm -rf $(VENV); \
	echo "Creating clean virtual environment..."; \
	$(UV) sync --locked --reinstall-package llm-sdk; \
	echo "Install to virtualenv $(VENV) Successful."

sdk-install:
	test -f $(SDK_ZIP) || { echo "Error: $(SDK_ZIP) not found." >&2; exit 1; }
	rm -rf llm_sdk
	unzip -q $(SDK_ZIP)

check-env:
	@test -x $(VENV)/bin/python && $(VENV)/bin/python -c 'import numpy, pydantic, torch' 2>/dev/null \
		|| { echo "Error: no valid virtualenv found ($(VENV)). Run 'make install' first." >&2; exit 1; }

run: check-env
	$(UV) run --no-sync python -m src $(FLAGS)

debug: check-env
	$(UV) run --no-sync python -m pdb -m src $(FLAGS)

clean:
	@set -eu; \
	find . -type d -name .git -prune -o -type d \
		\( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) \
		-prune -exec rm -rf {} +; \
	echo "Clean Successful."

test: check-env
	@if test -d tests; then $(UV) run --no-sync python -m unittest discover -s tests -v; \
	else echo "No tests: this checkout has no tests/ directory (tests are not part of the submission)."; fi

lint: check-env
	@result=0; $(UV) run --no-sync flake8 . || result=1; \
	$(UV) run --no-sync mypy . $(MYPY_FLAGS) || result=1; \
	exit $$result

lint-strict: check-env
	@result=0; $(UV) run --no-sync flake8 . || result=1; \
	$(UV) run --no-sync mypy . --strict || result=1; \
	exit $$result

-include dev.mk
