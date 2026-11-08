# Local overrides (git-ignored), e.g. `export HF_HOME := ...`.
-include .local.mk

UV ?= uv
FLAGS ?=
MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# CPU environment: what `uv sync` builds and what the reviewers use.
VENV ?= .venv
# Development-only GPU environment, kept apart so `make install` never touches it.
# The SDK picks mps > cuda > cpu itself.
GPU_ENV ?= .venv-gpu
# Linux with an AMD GPU (/dev/kfd) gets the ROCm build of torch; macOS needs nothing extra.
ROCM_INDEX ?= https://download.pytorch.org/whl/rocm7.2

GPU_UV = UV_PROJECT_ENVIRONMENT=$(GPU_ENV) $(UV)

# $(call require_env,<venv>,<make target that builds it>)
# Stops with a clear message when the environment is missing or cannot import
# the core packages (torch included: nothing runs without it), so a broken
# environment is never used silently. The SDK itself is not imported here.
define require_env
@test -x $(1)/bin/python && $(1)/bin/python -c 'import numpy, pydantic, torch' 2>/dev/null \
	|| { echo "Error: no valid virtualenv found ($(1)). Run 'make $(2)' first." >&2; exit 1; }
endef

.PHONY: help install gpu-install sdk-install run debug gpu-run gpu-debug clean test \
	lint lint-strict subject subject-audit

help:
	@echo "Available targets:"
	@echo "  make install       Recreate $(VENV) from scratch (CPU torch, from uv.lock)"
	@echo "  make gpu-install   Recreate $(GPU_ENV) from scratch (GPU torch + accelerate)"
	@echo "  make sdk-install   Unpack llm_sdk from llm_sdk.zip (or SDK_ZIP=<zip>)"
	@echo "  make run           Run the project"
	@echo "  make debug         Run the project with pdb"
	@echo "  make gpu-run       Run the project in $(GPU_ENV)"
	@echo "  make gpu-debug     Run the project in $(GPU_ENV) with pdb"
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

gpu-install:
	@set -eu; \
	rm -rf $(GPU_ENV); \
	echo "Creating clean GPU virtual environment..."; \
	$(GPU_UV) sync --locked --reinstall-package llm-sdk; \
	if [ -e /dev/kfd ]; then \
		echo "AMD GPU found: installing the ROCm build of torch..."; \
		$(UV) pip install --python $(GPU_ENV)/bin/python --reinstall-package torch --index $(ROCM_INDEX) torch; \
	fi; \
	$(UV) pip install --python $(GPU_ENV)/bin/python accelerate; \
	echo "Install to virtualenv $(GPU_ENV) Successful."

# Replace llm_sdk with the contents of a zip (default: the distributed one).
# Another zip: `make sdk-install SDK_ZIP=path/to/other.zip` (same llm_sdk/ layout).
SDK_ZIP ?= llm_sdk.zip
sdk-install:
	test -f $(SDK_ZIP) || { echo "Error: $(SDK_ZIP) not found." >&2; exit 1; }
	rm -rf llm_sdk
	unzip -q $(SDK_ZIP)

run:
	$(call require_env,$(VENV),install)
	$(UV) run --no-sync python -m src $(FLAGS)

debug:
	$(call require_env,$(VENV),install)
	$(UV) run --no-sync python -m pdb -m src $(FLAGS)

gpu-run:
	$(call require_env,$(GPU_ENV),gpu-install)
	$(GPU_UV) run --no-sync python -m src $(FLAGS)

gpu-debug:
	$(call require_env,$(GPU_ENV),gpu-install)
	$(GPU_UV) run --no-sync python -m pdb -m src $(FLAGS)

clean:
	find . -type d \( -name .venv -o -name .venv-gpu -o -name .git -o -name llm_sdk \) -prune -o -type d \( -name __pycache__ -o -name .mypy_cache \) -exec rm -rf {} +

test:
	$(call require_env,$(VENV),install)
	$(UV) run --no-sync python -m unittest discover -s tests -v

# Subject IV.2 commands, verbatim. llm_sdk is excluded by .flake8 and pyproject.toml.
# Run both even if flake8 fails, and preserve a failure status.
lint:
	$(call require_env,$(VENV),install)
	@result=0; $(UV) run --no-sync flake8 . || result=1; \
	$(UV) run --no-sync mypy . $(MYPY_FLAGS) || result=1; \
	exit $$result

lint-strict:
	$(call require_env,$(VENV),install)
	@result=0; $(UV) run --no-sync flake8 . || result=1; \
	$(UV) run --no-sync mypy . --strict || result=1; \
	exit $$result

subject:
	$(UV) run --no-sync python -m tools.check_subject
	@if test -d src; then $(UV) run --no-sync python -m src --help; fi

# Fails on the PoC. This is a preflight, never a complete compliance certificate.
subject-audit:
	$(UV) run --no-sync python -m tools.check_subject --submission
	$(MAKE) lint
