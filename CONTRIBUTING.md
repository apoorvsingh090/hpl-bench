# Contributing to HPL-Bench

HPL-Bench measures Hindi **reading (पठन)** and **writing (लेखन)** — not just triple extraction.

## What we accept

* Native-written Hindi items (no raw MT). If you adapt from NCERT / public sources, note `source` and license.
* One item = one JSON line matching `schema/hpl_schema.json`.
* Every item needs: `id`, `track`, `level`, `prompt_hi`, `answer` or `rubric`, `source`.
* Passages should be 150-800 words for RC, shorter for grammar.

## Levels

* `A1/A2` — primary literacy, simple sentences
* `B1/B2` — secondary / senior-secondary (NCERT 8-12)
* `C1` — advanced / sahityik / sarkari / multi-doc reasoning

## Tracks

| track | meaning |
|---|---|
| `padh` | passage comprehension (MCQ + short + long) |
| `anumaan` | inference, NLI, coref, muhavare/lokokti |
| `vyakaran` | error correction, minimal pairs, karak/ling-vachan |
| `saar` | summarization / paraphrase with grounding |
| `lekhan` | essay, letter, samvad, suchna (rubric-scored) |
| `register` | tat-sama / urdu-nishth / Hinglish robustness |
| `grounded` | long / multi-doc QA with citation, anti-hallucination |

## Quality bar

1. `python src/validate.py --data data/dev/hpl_dev.jsonl` must pass.
2. No duplicate prompts (normalized whitespace check).
3. Distractors for MCQ must be plausible, single correct answer.
4. Writing prompts must include a 1-5 rubric (`shuddhata, spashtata, sangathan, upyuktata`).
5. Avoid religious/political flamebait. Folk, science, history, civic topics preferred.

## Adding items

```powershell
python src/validate.py --data data/dev/hpl_dev.jsonl
python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds examples/sample_preds.jsonl --split auto
```

Hidden test (`data/test_hidden/`) is maintained by maintainers only — do not PR answers for it.
