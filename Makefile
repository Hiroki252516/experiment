PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
MODEL ?= gemma3:1b
BASE_URL ?= http://localhost:11434

.PHONY: setup test smoke run analyze

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/smoke_test.py --model $(MODEL) --base-url $(BASE_URL)

run:
	$(PYTHON) scripts/run_experiment.py --episodes 10 --conditions comm silent random --model $(MODEL) --base-url $(BASE_URL)

analyze:
	$(PYTHON) scripts/analyze_results.py --input logs/results.csv --figure logs/summary.png
