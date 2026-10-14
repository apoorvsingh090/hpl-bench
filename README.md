# HPL-Bench — Hindi Pathan-Lekhan Bench

**Hindi reading + writing comprehension for AI models.**

MIT-licensed, open, native-Hindi-first. Built as a modern successor to
[`ritwikmishra/hindi-benchie`](https://github.com/ritwikmishra/hindi-benchie)
(112-sentence OIE triple bench, IJCNLP-AACL 2023), which is excellent for
Open IE but insufficient for LLM literacy: single sentences only, exact-match
triple scoring, no passages, no writing, no Hinglish/register, no grounding.

HPL-Bench instead tests whether a model can **read like a Class 8–12 student
and write like a competent adult** in Hindi.

## Tracks (v0.1 seed: 50 items, target: 3000)

| track | Hindi name | what it tests | scoring |
|---|---|---|---|
| `padh` | पठन-बोध | 150–800 word passages: tathya / anumaan / vivechan QA | MCQ accuracy + short-answer F1 |
| `anumaan` | अनुमान-तर्क | NLI, coref (`वह/उसने/उसे`), muhavare/lokokti | accuracy |
| `vyakaran` | व्याकरण-शुद्धि | karak, ling-vachan, kaal, matra, anusvaar, sandhi | exact + token-F1 |
| `saar` | सार-लेखन | summarization/paraphrase, no hallucination | chrF/BERTScore + LLM-judge |
| `lekhan` | लेखन | nibandh, patra, samvad, suchna (60–200 words) | rubric 1–5 LLM-judge + human audit |
| `register` | प्रयुक्ति | tat-sama / Urdu-nishth / Hinglish, `आप/तुम/तू` | accuracy + style-appropriateness |
| `grounded` | प्रमाणिक QA | multi-doc, cite `[अंश 1]`, say `जानकारी उपलब्ध नहीं है` when absent | citation-F1 + abstention rate |

Levels: `A1/A2` primary, `B1/B2` secondary (NCERT 8–12), `C1` advanced/sahityik/sarkari.

Related work we complement (not duplicate): MILU, IndicGenBench (XQuAD-IN/XORQA-IN),
IndicMMLU-Pro, IFEval-Hi / MT-Bench-Hi / GSM8K-Hi / ChatRAG-Hi, HindiRC.

## Quickstart

```powershell
# validate schema
python src/validate.py --data data/dev/hpl_dev.jsonl

# score your model (preds: {"id":..., "prediction":...} per line)
python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds examples/sample_preds.jsonl
```

Prompts must be in Hindi — see `examples/prompts_hi.md`.
Contributing: `CONTRIBUTING.md`. Hidden test: `data/README.md`.

## Repo layout

```
hpl-bench/
  LICENSE  MIT
  README.md
  schema/hpl_schema.json
  data/dev/hpl_dev.jsonl        # public dev (50 seed)
  data/test_hidden/             # git-ignored, leaderboard only
  src/validate.py               # stdlib schema check
  src/evaluate.py               # stdlib scorer (auto part)
  examples/prompts_hi.md
  examples/sample_preds.jsonl
```

## Scoring policy

* Auto (deterministic): MCQ/vyakaran accuracy, short-answer token-F1.
* Judge (semantic): `saar/lekhan/grounded-long` use Hindi rubric
  (`शुद्धता, स्पष्टता, संगठन, उपयुक्तता` 1–5) with LLM-judge + 5% human audit.
  `evaluate.py` reports token-F1 as informative-only for those tracks.
* Report per-track + Pathan vs Lekhan split + `shuddh vs Hinglish` robustness delta.
* `dev-public 30% / test-hidden 70%`. Version yearly to limit contamination.

## License

MIT — see `LICENSE`. Data seeds are original examples (`seed:v0.1`).
If you contribute NCERT-derived items, note source + license in `source`.
