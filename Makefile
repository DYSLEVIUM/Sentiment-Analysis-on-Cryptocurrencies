.PHONY: install run

run: install
	python src.main.py

install:
	pip install -r requirements.txt
	pip install -e .
