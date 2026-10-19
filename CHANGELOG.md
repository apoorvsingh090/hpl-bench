# Changelog

All notable changes to HPL-Bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Git tags `v0.1.0` / `v0.1.1` mark the releases below.

## [Unreleased]

### Added

- GitHub Actions CI: schema validation + sample-pred scoring smoke test (stdlib)
- CodeQL code scanning (Python)
- Community files: `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`
- Issue forms (bug report, feature request) and pull request template
- `dependabot.yml` (weekly grouped GitHub Actions updates)

## [0.1.1] - 2026-10-14

### Fixed

- v0.1.1 audit fixes: dataset answers, schema, scorer/validator/runner
  robustness, judge spec, canary
- Hindi polish: VY-003 explanation, SA-002 agreement, SA-004 number
- MICRO-AUTO > 1.0 latent bug and duplicate prediction IDs

### Added

- `src/run_local.py`: local Ollama / HF-transformers / vLLM runner (stdlib-first)
- `src/run_hf_api.py`: Hugging Face Inference Providers runner (no local GPU)

## [0.1.0] - 2026-10-14

### Added

- Initial HPL-Bench: Hindi Pathan-Lekhan Bench, 50 seed dev items
- Tracks: comprehension, inference, grammar/agreement, summarization,
  writing (rubric-scored), register/romanization, grounded QA with citations
- Stdlib validator (`src/validate.py`) and scorer (`src/evaluate.py`)
- Schema (`schema/hpl_schema.json`), judge rubric, prompts, sample preds,
  contamination canary, CITATION.cff, MIT license
