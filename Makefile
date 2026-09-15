.PHONY: sync test test-core test-native test-reference reference he lint build paper whitepaper docs check clean
UV ?= uv
PAPER_PYTHON ?= 3.13

sync:
	bash scripts/sync_environment.sh
test-core:
	cargo test -p pllm-core
test-native:
	PLLM_REQUIRE_RUST=1 uv run pytest -m rust
test-reference:
	PLLM_KERNEL_BACKEND=python uv run pytest -m "not rust"
test:
	uv run pytest -m 'not he'
reference:
	uv run python scripts/generate_developer_reference.py --check
he:
	uv run --extra he pytest -m he
lint:
	uv run ruff check python/pllm scripts tests
build:
	uv build
paper:
	$(UV) run --no-project --python $(PAPER_PYTHON) python scripts/build_paper.py
whitepaper:
	$(UV) run --no-project --python $(PAPER_PYTHON) python scripts/build_whitepaper.py --publish
docs: paper whitepaper
	$(UV) run python scripts/generate_developer_reference.py
	$(UV) run --no-project --python $(PAPER_PYTHON) python scripts/prepare_docs.py
	cd docs && npm install && npm test && npm run typecheck && npm run build
check: reference
	$(UV) run --no-project --python $(PAPER_PYTHON) python scripts/check_repository.py
clean:
	$(MAKE) -C paper clean
	rm -f paper/whitepaper.pdf paper/whitepaper-source.zip
	rm -rf build dist docs/.next docs/out .pytest_cache
