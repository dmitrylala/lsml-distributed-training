PYTHON?=uv run python

FMT_DIRS=common scripts


.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-17s\033[0m %s\n", $$1, $$2}'

vendor: ## Install all necessary packages
	uv sync

fmt: vendor ## Format code
	@echo === FMT ===
	# Fix linting violations
	$(PYTHON) -m ruff check --fix $(FMT_DIRS) || true
	# Full code formatting
	$(PYTHON) -m ruff format $(FMT_DIRS)

lint-ruff:
	@echo === LINT RUFF ===
	$(PYTHON) -m ruff check --output-format=pylint $(FMT_DIRS)

lint-mypy:
	@echo === LINT MYPY ===
	$(PYTHON) -m mypy \
		--skip-cache-mtime-checks \
		--exclude .*\\.ipynb \
		$(FMT_DIRS)

lint: vendor lint-ruff lint-mypy ## Lint code

clean: ## Clean generated files
	@find . -name '*.py[cod]' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +
	@find . -name '*$py.class' -exec rm -rf {} +

# Prevent make from trying to build .py files as targets
.PHONY: $(filter %.py,$(MAKECMDGOALS))
$(filter %.py,$(MAKECMDGOALS)):
	@:
