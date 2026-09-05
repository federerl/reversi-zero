.DEFAULT_GOAL := help
SHELL := /bin/bash

UV      ?= uv
RUN     ?= $(UV) run
TORCH   ?= cpu          # `make TORCH=cu124 setup` on the GPU cluster
CONFIG  ?= configs/dev8x8.yaml

.PHONY: help setup test test-all quality fmt smoke train-dev train-full bench arena calibrate demo plots clean

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- environment -----------------------------------------------------------

setup:  ## Install Python + web dependencies (TORCH=cpu|cu124)
	$(UV) sync --extra $(TORCH) --extra dev --extra api --extra obs
	@test -f web/package.json && npm --prefix web ci || echo "web/ not present yet; skipping npm"

# --- quality gates ---------------------------------------------------------

quality:  ## Lint, format check, and type check
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) pyright

fmt:  ## Auto-fix lint and formatting
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test:  ## Fast test selection (CPU, no GPU) -- target < 90s
	$(RUN) pytest -m "not slow and not gpu"

test-all:  ## Everything, including the 50k-playout differential and integration tests
	$(RUN) pytest -m "not gpu"

# --- training --------------------------------------------------------------

smoke:  ## 4x4 end-to-end learning gate on CPU (~10 min)
	$(RUN) reversi train -c configs/smoke4x4.yaml

train-dev:  ## 8x8 development run (~60-90 min on one GPU)
	$(RUN) reversi train -c configs/dev8x8.yaml --resume auto

train-full:  ## 8x8 overnight run; resumes across the 8h job ceiling
	$(RUN) reversi train -c configs/full8x8.yaml --resume auto

# --- measurement -----------------------------------------------------------

bench:  ## Engine / MCTS / inference / self-play throughput -> bench/results/
	$(RUN) reversi bench -c $(CONFIG)

arena:  ## Rate a run's checkpoints against the frozen baselines (RUN_ID=<run id>)
	$(RUN) reversi arena -c $(CONFIG) --suite crossgen --run-id $(RUN_ID) --workers 8

calibrate:  ## Search and validate the four difficulty levels
	$(RUN) reversi calibrate -c $(CONFIG) --validate

plots:  ## Regenerate every figure from the JSONL metric streams
	$(RUN) python scripts/make_plots.py

# --- product ---------------------------------------------------------------

demo:  ## Fetch the released checkpoint, build the web app, and serve it
	$(RUN) python scripts/download_model.py
	npm --prefix web run build
	$(RUN) reversi serve

clean:  ## Remove caches (never touches runs/ or models/)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
