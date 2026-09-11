.PHONY: all pdf arxiv web clean
UV ?= uv
PYTHON_VERSION ?= 3.13

all:
	$(UV) run --no-project --python $(PYTHON_VERSION) python ../scripts/build_paper.py
pdf:
	$(UV) run --no-project --python $(PYTHON_VERSION) python ../scripts/build_paper.py pdf
arxiv:
	$(UV) run --no-project --python $(PYTHON_VERSION) python ../scripts/build_paper.py arxiv
web:
	$(UV) run --no-project --python $(PYTHON_VERSION) python ../scripts/build_paper.py web
clean:
	rm -f main.pdf main.tex build-metadata.json arxiv-source.tar.gz
