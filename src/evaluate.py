"""HPL-Bench evaluator (stdlib only, no hard dependencies).

Usage:
  python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds examples/sample_preds.jsonl

Preds format (JSONL, one per line):
  {"id": "HPL-PA-001", "prediction": " ... "}

Scoring:
  - MCQ / single-answer (choices present): normalized exact-match accuracy.
  - Short answer (vyakaran, anumaan, register, padh-short, grounded-short):
    exact-match + whitespace token-F1 (informative).
  - Long-form (saar, lekhan, grounded-long): token-F1 vs reference is
    informative only; flagged as needs_human_or_llm_judge.
    Use an LLM-judge with the Hindi rubric in README for official scores.
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

STRIP_CHARS = " \t\n\r'\"“”‘’।.,!?;:()[]{}-–—"

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.strip().strip(STRIP_CHARS).strip()
    s = re.sub(r"\s+", " ", s)
    # casefold only helps Hinglish / Latin; harmless for Devanagari
    if re.search(r"[A-Za-z]", s):
        s = s.casefold()
    return s

def tokenize(s: str):
    s = normalize(s)
    toks = []
    for t in s.split(" "):
        t = t.strip(STRIP_CHARS)
        if t:
            toks.append(t)
    return toks

def token_f1(pred: str, gold: str) -> float:
    pt, gt = tokenize(pred), tokenize(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    from collections import Counter
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pt)
    r = overlap / len(gt)
    return 2 * p * r / (p + r)

def is_long_form(item) -> bool:
    if item.get("track") in ("saar", "lekhan"):
        return True
    # grounded items with long answers (>25 tokens) need judge
    if item.get("track") == "grounded" and len(tokenize(item.get("answer", ""))) > 25:
        return True
    return False

def score_item(item, pred: str):
    gold = item.get("answer", "")
    if not gold.strip():
        # rubric-scored (e.g. lekhan prompts with no reference answer):
        # auto exact/F1 are meaningless — defer to LLM-judge + human audit.
        return {"exact": 0, "token_f1": 0.0, "skipped_auto": True}
    acc_answers = [gold] + item.get("acceptable_answers", [])
    norm_pred = normalize(pred)
    exact = any(norm_pred == normalize(a) for a in acc_answers)
    f1 = max(token_f1(pred, a) for a in acc_answers) if acc_answers else 0.0
    return {"exact": int(exact), "token_f1": round(f1, 4), "skipped_auto": False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    gold = {}
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            gold[o["id"]] = o

    preds = {}
    with open(args.preds, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            preds[o["id"]] = o.get("prediction", o.get("answer", ""))

    per_track = defaultdict(lambda: {"n": 0, "exact": 0, "f1_sum": 0.0, "judge_needed": 0})
    rows = []
    for _id, item in sorted(gold.items()):
        pred = preds.get(_id, "")
        s = score_item(item, pred)
        tr = item["track"]
        per_track[tr]["n"] += 1
        per_track[tr]["exact"] += s["exact"]
        per_track[tr]["f1_sum"] += s["token_f1"]
        needs_judge = is_long_form(item)
        if needs_judge:
            per_track[tr]["judge_needed"] += 1
        rows.append({"id": _id, "track": tr, **s,
                     "needs_judge": needs_judge,
                     "missing_pred": _id not in preds})

    print(f"{'track':<10}{'n':>4}  {'exact_acc':>10}  {'avg_tokenF1':>12}  judge_needed")
    total_n = total_exact = 0
    total_f1 = 0.0
    for tr in sorted(per_track):
        d = per_track[tr]
        acc = d["exact"] / d["n"] if d["n"] else 0
        avgf1 = d["f1_sum"] / d["n"] if d["n"] else 0
        print(f"{tr:<10}{d['n']:>4}  {acc:>10.3f}  {avgf1:>12.3f}  {d['judge_needed']}")
        total_n += d["n"]
        total_exact += d["exact"]
        total_f1 += d["f1_sum"]
    if total_n:
        print("-" * 52)
        print(f"{'MICRO-ALL':<10}{total_n:>4}  {total_exact/total_n:>10.3f}  {total_f1/total_n:>12.3f}")
        auto_n = sum(v["n"] - v["judge_needed"] for v in per_track.values())
        auto_exact = sum(v["exact"] for k, v in per_track.items() if k not in ("saar", "lekhan"))
        # note: grounded-long excluded from auto accuracy in official reporting
        print(f"\nNote: saar/lekhan (+ grounded-long) are reference-only here.")
        print(f"Official score for those tracks = LLM-judge (1-5 rubric) + 5% human audit.")
        print(f"Missing preds: {sum(1 for _id in gold if _id not in preds)}/{total_n}")

    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows,
            "per_track": {k: {"n": v["n"], "acc": v["exact"]/v["n"] if v["n"] else 0,
            "avg_f1": v["f1_sum"]/v["n"] if v["n"] else 0} for k, v in per_track.items()}},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
