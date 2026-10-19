# HPL-Bench — Hindi Pathan-Lekhan Bench

[![CI](https://img.shields.io/github/actions/workflow/status/apoorvsingh090/hpl-bench/ci.yml?branch=master)](https://github.com/apoorvsingh090/hpl-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-stdlib%20only-blue.svg)](src)

**Hindi reading + writing comprehension for AI models.**

MIT-licensed, open, native-Hindi-first. Built as a modern successor to
[`ritwikmishra/hindi-benchie`](https://github.com/ritwikmishra/hindi-benchie)
(112-sentence OIE triple bench, IJCNLP-AACL 2023), which is excellent for
Open IE but insufficient for LLM literacy: single sentences only, exact-match
triple scoring, no passages, no writing, no Hinglish/register, no grounding.

HPL-Bench instead tests whether a model can **read like a Class 8–12 student
and write like a competent adult** in Hindi.

## Tracks (v0.1.1 seed: 50 items, target: 3000)

| track | Hindi name | what it tests | scoring |
|---|---|---|---|
| `padh` | पठन-बोध | passages: tathya / anumaan / vivechan QA (seed: short passages; target 150–800 words) | MCQ accuracy + short-answer F1 |
| `anumaan` | अनुमान-तर्क | NLI, coref (`वह/उसने/उसे`), muhavare/lokokti | accuracy |
| `vyakaran` | व्याकरण-शुद्धि | karak, ling-vachan, kaal, matra, anusvaar, sandhi | exact + token-F1 |
| `saar` | सार-लेखन | summarization/paraphrase, no hallucination (seed: short passages, distinct from padh) | LLM-judge + token-F1 informative |
| `lekhan` | लेखन | nibandh, patra, samvad, suchna (60–200 words) | rubric 1–5 LLM-judge + human audit |
| `register` | प्रयुक्ति | tat-sama / Urdu-nishth / Hinglish, `आप/तुम/तू` | accuracy + style-appropriateness |
| `grounded` | प्रमाणिक QA | 2-snippet toys in seed (target: multi-doc), cite `[अंश 1]`, say `जानकारी उपलब्ध नहीं है` when absent | accuracy + abstention rate; long-form via judge |

Levels in seed v0.1.1: `A1:6 / A2:3 / B1:19 / B2:15 / C1:7`. Target adds C1 sahityik/sarkari + long-doc.

Related work we complement (not duplicate): MILU, IndicGenBench (XQuAD-IN/XORQA-IN),
IndicMMLU-Pro, IFEval-Hi / MT-Bench-Hi / GSM8K-Hi / ChatRAG-Hi, HindiRC.

## Quickstart

```powershell
# validate schema
python src/validate.py --data data/dev/hpl_dev.jsonl

# score your model (preds: {"id":..., "prediction":...} per line)
python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds examples/sample_preds.jsonl
```

## Run it on your own local model — yes

`src/run_local.py` builds the Hindi prompts from `examples/prompts_hi.md`,
runs your local model, and writes the `preds.jsonl` that `evaluate.py` scores.
No data leaves your machine (except hidden-test, which you don't have).

```powershell
# 0. smoke test, no model needed
python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds.jsonl --backend dummy --limit 5
python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds preds.jsonl

# 1. Ollama — easiest local path (stdlib only, no pip deps)
# ollama pull gemma3:4b   # or aya-expanse:8b, qwen3:8b, etc.
python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds.jsonl --backend ollama --model gemma3:4b
python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds preds.jsonl

# 2. HuggingFace transformers — needs GPU for 2B+ models
# pip install transformers torch --upgrade
python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds.jsonl --backend hf --model google/gemma-2-2b-it --max-new-tokens 256

# 3. vLLM / LM Studio / llama.cpp server — any OpenAI-compatible endpoint
# vllm serve google/gemma-2-2b-it --port 8000
python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds.jsonl --backend openai-compat --model google/gemma-2-2b-it --base-url http://localhost:8000/v1
```

## No local GPU? Run on Hugging Face cloud

`src/run_hf_api.py` sends the same Hindi prompts to HF Inference Providers —
compute runs on HF servers, your machine only scores. Stdlib only, just needs a free token.

```powershell
# free account + token (read-only is enough)
# https://huggingface.co/join  →  https://huggingface.co/settings/tokens
$env:HF_TOKEN = "hf_..."

# smoke test first (free-tier rate limits), then full 50
python src/run_hf_api.py --gold data/dev/hpl_dev.jsonl --out preds.jsonl --model google/gemma-2-2b-it --limit 5
python src/run_hf_api.py --gold data/dev/hpl_dev.jsonl --out preds.jsonl --model google/gemma-2-2b-it
python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds preds.jsonl
```

Try Hindi-capable models: `google/gemma-2-2b-it`, `meta-llama/Llama-3.2-3B-Instruct`,
`Qwen/Qwen3-8B`, `sarvamai/sarvam-2b-v0.5`, `CohereLabs/aya-expanse-8b`.
Next step (optional): host the scorer as a Hugging Face Space so anyone can upload
`preds.jsonl` and get the table without cloning.

Notes:
* Prompts are fixed Hindi templates — same for every model, so runs are comparable.
  Exception: `register` items contain documented Hinglish/English spans by design.
* Deterministic scoring (`padh/anumaan/vyakaran/register` + short QA) runs fully offline.
  `evaluate.py` also prints `MICRO-AUTO` (official auto), `PATHAN` (padh+anumaan+grounded),
  `LEKHAN` (vyakaran+saar+lekhan), `REGISTER` aggregates. `avg_tokenF1` is informative only.
* `saar/lekhan` need a judge: see `examples/judge_rubric_hi.md`
  (`शुद्धता, स्पष्टता, संगठन, उपयुक्तता` 1–5, temp 0.0, kappa ≥0.6 target), or human-read a 5% sample.
* Dev set is 50 items (~5 min on a 4B model via Ollama). Full 3000-item bench will be sharded the same way.
* Preds use only the `prediction` key; a stray `answer` key is ignored so gold-as-preds cannot ace.
* Bare MCQ indices (`1`, `(b)`) resolving to the right choice count as correct.

Prompts must be in Hindi — see `examples/prompts_hi.md`.
Contributing: `CONTRIBUTING.md`. Hidden test: `data/README.md`.

## Repo layout

```
hpl-bench/
  LICENSE  MIT
  README.md
  CITATION.cff
  CONTRIBUTING.md
  requirements.txt
  .gitignore
  schema/hpl_schema.json
  data/dev/hpl_dev.jsonl        # public dev (50 seed v0.1.1)
  data/canary.txt               # contamination canary, keep out of training
  data/README.md
  data/test_hidden/             # git-ignored, leaderboard only
  src/validate.py               # stdlib schema check
  src/evaluate.py               # stdlib scorer (auto part + aggregates)
  src/run_local.py              # run Ollama / HF-transformers / vLLM locally -> preds.jsonl
  src/run_hf_api.py             # run on HF Inference Providers (no local GPU) -> preds.jsonl
  examples/prompts_hi.md
  examples/judge_rubric_hi.md   # LLM-judge spec + bias controls
  examples/sample_preds.jsonl
```

## Scoring policy

* Auto (deterministic): MCQ/vyakaran accuracy (exact, Hindi-insensitive:
  chandrabindu~anusvara, ZWJ/ZWNJ dropped). Short-answer token-F1 informative only.
* Judge (semantic): `saar/lekhan/grounded-long` use `examples/judge_rubric_hi.md`
  with LLM-judge + 5% human audit (10% for C1/lekhan suggested).
  `evaluate.py` reports token-F1 as informative-only for those tracks.
* Report per-track + `PATHAN` (padh+anumaan+grounded) vs `LEKHAN` (vyakaran+saar+lekhan)
  + `REGISTER` robustness. Seed `saar` passages are distinct from `padh`.
* `dev-public / test-hidden` (target 30/70). Canary in `data/canary.txt`.
  Version yearly to limit contamination. Fixed public templates can overfit —
  hidden test uses held-out passages.

## Limitations (seed)

* Passages are short toys (≤60 words), not 150–800; no tables/long-doc/dialect beyond Hinglish.
* No verifiable instruction-following / multi-turn / math / tool-use (see IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, BFCL-Hi).
* n=50 → per-track CIs ±15–30%. Don't leaderboard on seed alone.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for item rules, levels, tracks, and
the quality bar; [CHANGELOG.md](CHANGELOG.md) tracks releases. By
participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Use
the issue templates for bugs and feature requests; beginner-friendly issues
carry the
[`good first issue`](https://github.com/apoorvsingh090/hpl-bench/labels/good%20first%20issue)
label.

## Security

Report benchmark-integrity or tooling vulnerabilities (hidden-test leaks,
scorer exploits) privately via
[GitHub Security Advisories](https://github.com/apoorvsingh090/hpl-bench/security/advisories);
see [SECURITY.md](SECURITY.md).

## License

MIT — see `LICENSE`. Data seeds are original examples (`seed:v0.1.1`).
If you contribute NCERT-derived items, note source + license in `source`.
