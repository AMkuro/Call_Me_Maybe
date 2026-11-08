# Development-only targets, included by the Makefile with `-include dev.mk`.
# Not listed in submission-files.txt, so the submitted Makefile has none of it.

# Local overrides (git-ignored), e.g. `export HF_HOME := ...`.
-include .local.mk

# Development-only GPU environment, kept apart so `make install` never touches it.
# The SDK picks mps > cuda > cpu itself.
GPU_ENV ?= .venv-gpu
# Linux with an AMD GPU (/dev/kfd) gets the ROCm build of torch; macOS needs nothing extra.
ROCM_INDEX ?= https://download.pytorch.org/whl/rocm7.2

GPU_UV = UV_PROJECT_ENVIRONMENT=$(GPU_ENV) $(UV)

.PHONY: dev-help gpu-install gpu-run gpu-debug subject subject-audit

dev-help:
	@echo "Development targets (dev.mk):"
	@echo "  make gpu-install   Recreate $(GPU_ENV) from scratch (GPU torch + accelerate)"
	@echo "  make gpu-run       Run the project in $(GPU_ENV)"
	@echo "  make gpu-debug     Run the project in $(GPU_ENV) with pdb"
	@echo "  make subject       Check the layout against the Subject (tools/check_subject.py)"
	@echo "  make subject-audit Stricter preflight; fails while src is missing"

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

gpu-run:
	$(call require_env,$(GPU_ENV),gpu-install)
	$(GPU_UV) run --no-sync python -m src $(FLAGS)

gpu-debug:
	$(call require_env,$(GPU_ENV),gpu-install)
	$(GPU_UV) run --no-sync python -m pdb -m src $(FLAGS)

subject:
	$(UV) run --no-sync python -m tools.check_subject
	@if test -d src; then $(UV) run --no-sync python -m src --help; fi

# Fails on the PoC. This is a preflight, never a complete compliance certificate.
subject-audit:
	$(UV) run --no-sync python -m tools.check_subject --submission
	$(MAKE) lint
