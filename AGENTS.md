# Repository Guidelines

## Project Structure & Module Organization
The Flask entry point lives in `app.py`, with HTML templates in `templates/` and static assets collocated where each view expects them. Domain logic is split across `bookmarks/` (ingest + parsing), `pipelines/` (batch workflows), `db/` (schema + sessions), and `utils/` (shared helpers, including configuration and GPU image tooling). Command-line utilities land in `scripts/`, and data artifacts produced during experiments stay under `build/` or `test/` to avoid polluting source modules.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package in editable mode with base dependencies.
- `python -m db.schema` provisions or upgrades the PostgreSQL schema defined in `db/schema.py`.
- `python run_pipeline.py` orchestrates scrape → describe → embed; pass `--skip-*` flags (e.g., `--skip-image-descriptions`) to focus on a single stage.
- `python -m app` runs the Flask UI at `http://localhost:5000` for manual verification.
- Image descriptions now use Gemini 2.5 Flash via `python -m scripts.image_descriptions_gemini` (no local model install required).

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and module-level `logger` instances for observability. Prefer `snake_case` for functions, variables, and module names; reserve `PascalCase` for classes and pipeline orchestrators. Keep scripts import-safe by guarding main entry points with `if __name__ == "__main__":`. When touching shared helpers, add type hints for public functions and concise docstrings describing inputs, outputs, and notable side effects.

## Testing Guidelines
Executable diagnostics sit in `test/`. Run them directly (e.g., `python test/test_image_download.py`) to validate integrations against live services. New pytest-style unit tests should follow the `test_*.py` pattern and live beside similar modules or under `test/`. Aim for coverage over database access and network calls by mocking external services; when hitting the real database, point `utils/config.py` at a disposable schema, and clean up via `test/reset_database.py`.

## Commit & Pull Request Guidelines
Recent history (`git log`) favors short, imperative subject lines ("Adds image-only querying," "Err handling for access restricted media"). Match that tone, keep subjects ≤50 characters, and include a focused body when extra context is needed. Pull requests should explain workflow impact, note required environment changes, and link relevant issues or pipelines. Attach screenshots or CLI output when UI or ingestion behavior changes, and list manual verification steps so reviewers can reproduce your results quickly.

## Environment & Data Protection
Secrets stay out of source control; load credentials through environment variables consumed in `utils/config.py`. Use the provided `docker-compose.yml` or local Postgres with the pgvector extension when developing features that touch embeddings. Large media fixtures belong in transient storage (`build/` or a local bucket), not the repository, to keep clone times manageable.
