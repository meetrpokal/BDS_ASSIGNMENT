# Contributing

## Workflow

1. Pull the latest `main`: `git pull --rebase origin main`
2. Work on your own branch, for example `feature/model`, `feature/dashboard`, `feature/docs-tests`.
3. Make small commits, each covering one logical change.
4. Push at least once a day and open a pull request into `main`.
5. A teammate reviews the pull request before it is merged.

## Commit messages

Use a short prefix followed by a summary in the imperative mood:

| Prefix | Use for |
|---|---|
| `feat:` | New functionality |
| `fix:` | Bug fixes |
| `refactor:` | Code changes without behavior changes |
| `perf:` | Performance improvements |
| `test:` | Adding or changing tests |
| `docs:` | Documentation and notebooks |
| `ci:` / `chore:` | Automation, tooling, housekeeping |

## Before pushing

```bash
pip install -r requirements-dev.txt
flake8 . --select=E9,F63,F7,F82
pytest
```

## Ownership

| Area | Owner |
|---|---|
| `ml_sentiment_model.py`, `config.py` | Member A |
| `ml_dashboard_complete.py` | Member B |
| `tests/`, `scripts/`, `notebooks/`, CI, docs | Member C |
