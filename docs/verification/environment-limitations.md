# Verification environment limitations

Recorded 2026-07-13.

- `uv run pytest` could not download dependencies because DNS/network access to PyPI is disabled in this sandbox. The project remains configured for `uv` in a normal environment.
- `ruff` and `mypy` are not installed in the available Python runtime and cannot be fetched here; their attempted command outputs are preserved alongside this note.
- The sandbox denies loopback socket connections (`curl: (7) failed to open socket: Operation not permitted`). Uvicorn did complete its application startup and shutdown lifecycle; see `api-start.txt`. Domain end-to-end behavior, including safe, approval/resume, and block branches, was verified in-process.
- `.git` is mounted read-only, so `git add`/`git commit` fail creating `index.lock`. No commit could be created in this execution environment.
