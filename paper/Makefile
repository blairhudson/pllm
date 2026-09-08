UV ?= uv
PYTHON_VERSION ?= 3.13

all:
	$(UV) run --no-project --python $(PYTHON_VERSION) python ../scripts/build_paper.py
clean:
	rm -f main.pdf
