# Lab04 synthetic network-data generator

Instructor-side scripts that batch-generate synthetic network-security data
(via the OpenAI API) ahead of class, for use across Lab04 (Bayesian
Networks / HMM) and later ML/DL/LLM labs. Students never run these scripts
or touch an API key here - they only load the generated data files.

## Environment

Uses the **repo-root** `.venv` (`CISC339/pyproject.toml`), not a local one:

```bash
cd /Users/tianyu/Notebooks/CISC339
uv sync
.venv/bin/python3 Lab04/data_gen/<script>.py ...
```

## API key

Add to the repo-root `.env` (not committed):

```
OPENAI_API_KEY=sk-...
```

This is a direct OpenAI account key (platform.openai.com), separate from
the `OPENROUTER_API_KEY` used by Lab03.

## Files

- `schemas.py` - pydantic schemas for the three data layers (flow-level
  tabular, session-level sequences, raw log text). See its docstring.
- `data/` - generated output (gitignored except for small samples).
