.PHONY: install test lint format coverage train generate serve app

PYTHON ?= python3
PIP ?= pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check .
	mypy

format:
	ruff format .
	ruff check --fix .

coverage:
	pytest tests/ --cov --cov-report=term-missing

train:
	$(PYTHON) scripts/train.py --config $(or $(CONFIG),configs/tiny.yaml)

generate:
	$(PYTHON) scripts/generate.py $(ARGS)

serve:
	@echo "Not implemented — see phases/17_phase13_inference_api.md"

app:
	streamlit run app/playground.py $(ARGS)
