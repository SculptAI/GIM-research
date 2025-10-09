install:
	uv sync --all-groups

serve:
	@if [ -z "$(model_path)" ]; then \
		echo "make serve model_path=/path/to/model"; \
		exit 1; \
	fi
	uv run vllm serve $(model_path) --max_model_len 12288

lint:
	uv run ruff check
	uv run ruff format --diff

lint-fix:
	uv run ruff check --fix
	uv run ruff format

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .ruff_cache
	rm -rf unsloth_compiled_cache
