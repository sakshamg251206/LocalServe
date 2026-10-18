# Python 3.12 + uv. `make check` is what CI runs (offline: no model, no network).
RUN ?= minimum
UV := uv run

.PHONY: install lint typecheck test integration check build-llama models sweep quality report

install:
	uv sync --frozen

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

typecheck:
	$(UV) mypy

test:
	$(UV) pytest -q -m "not integration"

integration:  # needs .vendor/llama.cpp build + models/ (run `make build-llama models`)
	$(UV) pytest -q -m integration

check: lint typecheck test

build-llama:
	./scripts/build_llama.sh

models:
	$(UV) python scripts/fetch_models.py

sweep:  # hours; resumable
	$(UV) python scripts/run_sweep.py --run-id $(RUN) --phase capacity
	$(UV) python scripts/make_report.py --run-id $(RUN)
	$(UV) python scripts/run_sweep.py --run-id $(RUN) --phase p70
	$(UV) python scripts/run_sweep.py --run-id $(RUN) --phase overhead

quality:
	$(UV) python scripts/run_quality.py --run-id $(RUN)

report:  # regenerates summary.json, figures and doc tables from committed raw results
	$(UV) python scripts/make_report.py --run-id $(RUN)
