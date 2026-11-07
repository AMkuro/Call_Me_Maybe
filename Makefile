UV ?= uv
FLAGS ?=
MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean test lint lint-app lint-sdk subject subject-audit

install:
	$(UV) sync --locked

run:
	$(UV) run --no-sync python -m src $(FLAGS)

debug:
	$(UV) run --no-sync python -m pdb -m src $(FLAGS)

clean:
	find . -type d \( -name .venv -o -name .git -o -name llm_sdk \) -prune -o -type d \( -name __pycache__ -o -name .mypy_cache \) -exec rm -rf {} +

test:
	$(UV) run --no-sync python -m unittest discover -s tests -v

# Subject IV.2 commands, including the unmodified distributed SDK.
# Run both even if flake8 fails, and preserve a failure status.
lint:
	@result=0; $(UV) run --no-sync flake8 . || result=1; \
	$(UV) run --no-sync mypy --config-file .github/ci/sdk-mypy.ini . $(MYPY_FLAGS) || result=1; \
	exit $$result

# Development gate: explicitly exclude vendor code, which lint-sdk audits.
lint-app:
	$(UV) run --no-sync flake8 . --extend-exclude .venv,.cache,.mypy_cache,llm_sdk
	$(UV) run --no-sync mypy . $(MYPY_FLAGS)

lint-sdk:
	$(UV) run --no-sync python -m tools.audit_sdk

subject:
	$(UV) run --no-sync python -m tools.check_subject
	@if test -d src; then $(UV) run --no-sync python -m src --help; fi

# Fails on the PoC. This is a preflight, never a complete compliance certificate.
subject-audit:
	$(UV) run --no-sync python -m tools.check_subject --submission
	$(MAKE) lint
