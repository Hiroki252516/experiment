PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
MODEL ?= gemma3:1b
BASE_URL ?= http://localhost:11434
COMM_PHASE_STEPS ?= 2
HARD_SPLIT_PROB ?= 0.5
MEMORY_BUDGET ?= 50

.PHONY: setup test smoke run analyze viewer

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/smoke_test.py --model $(MODEL) --base-url $(BASE_URL)

run:
	$(PYTHON) scripts/run_experiment.py --episodes 50 --conditions comm silent random --model $(MODEL) --base-url $(BASE_URL) --comm-phase-steps $(COMM_PHASE_STEPS) --randomize-positions --hard-split-prob $(HARD_SPLIT_PROB) --memory-budget $(MEMORY_BUDGET)

analyze:
	$(PYTHON) scripts/analyze_results.py --input logs/results.csv --trace-dir logs/traces --figure logs/summary.png

viewer:
	$(PYTHON) -m streamlit run viewer/app.py
