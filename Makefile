.PHONY: install setup-data run test

install:
	pip install -r requirements.txt
	pip install -e .

setup-data:
	python scripts/ensure_ml_datasets.py

run:
	streamlit run app/streamlit_app.py

test:
	pytest tests/
