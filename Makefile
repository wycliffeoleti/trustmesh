.PHONY: run test eval lint typecheck
run:
	uv run uvicorn trustmesh.app:app --reload
test:
	uv run pytest -q
eval:
	uv run python evals/run_eval.py
lint:
	uv run ruff check .
typecheck:
	uv run mypy trustmesh
