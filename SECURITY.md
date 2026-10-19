# Security Policy

## Reporting a vulnerability

Report **privately**, never as a public issue:

1. Preferred: GitHub Security Advisories —
   **Security tab → Report a vulnerability**
   (https://github.com/apoorvsingh090/hpl-bench/security/advisories/new).
2. Otherwise contact [@apoorvsingh090](https://github.com/apoorvsingh090)
   via GitHub's private contact options.

## Scope

In scope:

- **Benchmark integrity:** anything that leaks the hidden test set
  (`data/test_hidden/` is git-ignored for a reason), the canary
  (`data/canary.txt`), or gold answers for held-out items
- **Scorer/validator exploits:** e.g. a prediction format that makes
  `src/evaluate.py` over-credit, path handling in `src/validate.py`
- **Runner issues:** unsafe handling in `src/run_local.py` /
  `src/run_hf_api.py` (e.g. command construction, token handling)
- Secrets accidentally committed (HF tokens, API keys)

Out of scope:

- Model performance on the benchmark itself
- Disagreements with individual gold answers (use a **Bug report** with
  evidence — those are content fixes, not security issues)

## Design notes

- Core eval is stdlib-only; optional backends are commented out in
  `requirements.txt`.
- The canary string in `data/canary.txt` exists to detect training-data
  contamination — please don't strip it from copies.

## Supported versions

Only the latest code on `master` (currently `v0.1.1`) is supported.
