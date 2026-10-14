# `data/` layout

* `data/dev/hpl_dev.jsonl` — public dev set (v0.1: 50 seed items). Use for development.
* `data/test_hidden/` — **not in git** (see `.gitignore`). Reserved for hidden test.
  Maintainers keep a 70% held-out set with same schema and track distribution.
  Leaderboard evaluates on hidden test to prevent contamination.

Each line is one item, UTF-8, `\n`-terminated JSON matching `schema/hpl_schema.json`.

Example:

```json
{"id": "HPL-VY-001", "track": "vyakaran", "level": "B1", "prompt_hi": "शुद्ध करो: लड़की ने किताब पढ़ा।", "answer": "लड़की ने किताब पढ़ी।", "explanation_hi": "कर्ता स्त्रीलिंग + कर्मणी प्रयोग में क्रिया कर्म के अनुसार।", "source": "seed:v0.1", "version": "0.1"}
```
