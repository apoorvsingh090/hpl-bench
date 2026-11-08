"""HPL-Bench evaluator (stdlib only, no hard dependencies).

Usage:
  python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds examples/sample_preds.jsonl

Preds format (JSONL, one per line):
  {"id": "HPL-PA-001", "prediction": " ... "}
  Only the "prediction" key is scored (a stray "answer" key is ignored).

Scoring:
  - MCQ / single-answer (choices present): normalized exact-match accuracy.
    A bare index ("1", "1.", "(b)") resolving to the correct choice counts.
  - Short answer: exact-match (official) + whitespace token-F1 (informative only;
    partial credit for wrong MCQ options must NOT be read as accuracy).
  - Long-form (saar, lekhan, grounded-long): token-F1 vs reference is
    informative only; flagged as needs_human_or_llm_judge.
    Official score = LLM-judge with examples/judge_rubric_hi.md + human audit.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from bench_common import (
    DEFAULT_STRIP,
    InputError,
    normalize,
    to_str,
    token_f1_tokens,
    tokenize,
)

# Hindi strip set: includes the danda; chandrabindu folds into anusvara.
STRIP_CHARS = DEFAULT_STRIP + "\u0964"
REPLACEMENTS = (("\u0901", "\u0902"),)
INDEX_RE = re.compile(r"^\(?\s*([1-5a-eA-E])\s*[.\)]?\s*$")

# Reading vs writing aggregates reported alongside per-track scores.
PATHAN_TRACKS = {"padh", "anumaan", "grounded"}
LEKHAN_TRACKS = {"vyakaran", "saar", "lekhan"}
REGISTER_TRACK = "register"


def normalize_text(s):
    return normalize(s, STRIP_CHARS, REPLACEMENTS)


def tokenize_text(s):
    return tokenize(s, STRIP_CHARS, REPLACEMENTS)


def token_f1(pred, gold):
    return token_f1_tokens(tokenize_text(pred), tokenize_text(gold))


def resolve_index(pred, choices):
    """Map a bare '1'/'b' style answer onto the referenced choice text."""
    if not choices:
        return pred
    m = INDEX_RE.match(to_str(pred).strip())
    if not m:
        return pred
    tok = m.group(1)
    idx = int(tok) - 1 if tok.isdigit() else (ord(tok.lower()) - ord("a"))
    if 0 <= idx < len(choices):
        return choices[idx]
    return pred


def prepare_gold(item):
    """Normalize the reference answers once per item (hot loop expects this)."""
    gold = to_str(item.get("answer", ""))
    aa = item.get("acceptable_answers", [])
    if not isinstance(aa, list):
        aa = []
    answers = [a for a in [gold] + aa if isinstance(a, str)]
    return {
        "gold": gold,
        "normalized_answers": frozenset(normalize_text(a) for a in answers),
        "answer_tokens": tuple(tokenize_text(a) for a in answers),
        "answer_token_count": len(tokenize_text(gold)),
    }


def is_long_form(item, prepared=None):
    if item.get("track") in ("saar", "lekhan"):
        return True
    if item.get("track") == "grounded":
        limit = prepared["answer_token_count"] if prepared else None
        return (limit if limit is not None else len(tokenize_text(item.get("answer", "")))) > 25
    return False


def score_item(item, pred, prepared=None):
    prepared = prepared or prepare_gold(item)
    pred = to_str(pred)
    if not prepared["gold"].strip():
        return {"exact": 0, "token_f1": 0.0, "skipped_auto": True}
    choices = item.get("choices") if isinstance(item.get("choices"), list) else None
    candidates = (pred, resolve_index(pred, choices))
    normalized_candidates = (normalize_text(c) for c in candidates)
    exact = int(any(c in prepared["normalized_answers"] for c in normalized_candidates))
    pred_tokens = tokenize_text(pred)
    f1 = round(max(
        [token_f1_tokens(pred_tokens, gt) for gt in prepared["answer_tokens"]] + [0.0]
    ), 4)
    return {"exact": exact, "token_f1": f1, "skipped_auto": False}


def load_jsonl(path, what):
    from bench_common import load_jsonl as _load

    try:
        return _load(path, what)
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)


def build_gold_map(gold_list):
    gold = {}
    for o in gold_list:
        if o["id"] in gold:
            print(f"ERROR: duplicate gold id '{o['id']}'")
            raise SystemExit(1)
        gold[o["id"]] = o
    return gold


def build_pred_map(pred_list):
    preds = {}
    for o in pred_list:
        if o["id"] in preds:
            print(f"ERROR: duplicate pred id '{o['id']}'")
            raise SystemExit(1)
        # NOTE: only "prediction" counts; a stray "answer" key is ignored
        # so that passing the gold file as preds cannot silently ace the run.
        preds[o["id"]] = o.get("prediction", "")
    return preds


def evaluate(gold, preds):
    """Score every gold item. Returns (per_track, rows)."""
    per_track = defaultdict(lambda: {"n": 0, "exact": 0, "f1_sum": 0.0,
                                     "auto_n": 0, "auto_exact": 0, "judge_needed": 0})
    rows = []
    for _id in sorted(gold):
        item = gold[_id]
        if "track" not in item:
            print(f"ERROR: gold id '{_id}' missing 'track'")
            raise SystemExit(1)
        prepared = prepare_gold(item)
        s = score_item(item, preds.get(_id, ""), prepared)
        tr = item["track"]
        d = per_track[tr]
        d["n"] += 1
        d["exact"] += s["exact"]
        d["f1_sum"] += s["token_f1"]
        needs_judge = is_long_form(item, prepared)
        d["judge_needed"] += int(needs_judge)
        if not needs_judge and not s.get("skipped_auto"):
            d["auto_n"] += 1
            d["auto_exact"] += s["exact"]
        rows.append({"id": _id, "track": tr, **s,
                     "needs_judge": needs_judge,
                     "missing_pred": _id not in preds})
    return per_track, rows


def aggregate(per_track):
    """Totals + official micro averages; MICRO-AUTO is always within [0, 1]."""
    total_n = sum(d["n"] for d in per_track.values())
    total_exact = sum(d["exact"] for d in per_track.values())
    total_f1 = sum(d["f1_sum"] for d in per_track.values())
    auto_n = sum(d["auto_n"] for d in per_track.values())
    auto_exact = sum(d["auto_exact"] for d in per_track.values())
    return {
        "total_n": total_n,
        "total_exact": total_exact,
        "total_f1": total_f1,
        "auto_n": auto_n,
        "auto_exact": auto_exact,
        "micro_all": (total_exact / total_n) if total_n else 0.0,
        "micro_auto": (auto_exact / auto_n) if auto_n else 0.0,
    }


def group_accuracy(per_track, names):
    n = sum(per_track[t]["n"] for t in names if t in per_track)
    e = sum(per_track[t]["exact"] for t in names if t in per_track)
    return n, (e / n if n else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    gold = build_gold_map(load_jsonl(args.gold, "gold"))
    preds = build_pred_map(load_jsonl(args.preds, "preds"))
    per_track, rows = evaluate(gold, preds)
    totals = aggregate(per_track)

    print(f"{'track':<10}{'n':>4}  {'exact_acc':>10}  {'avg_tokenF1*':>13}  judge_needed")
    for tr in sorted(per_track):
        d = per_track[tr]
        acc = d["exact"] / d["n"] if d["n"] else 0
        avgf1 = d["f1_sum"] / d["n"] if d["n"] else 0
        print(f"{tr:<10}{d['n']:>4}  {acc:>10.3f}  {avgf1:>13.3f}  {d['judge_needed']}")

    if totals["total_n"]:
        print("-" * 55)
        print(f"{'MICRO-ALL':<10}{totals['total_n']:>4}  {totals['micro_all']:>10.3f}"
              f"  {totals['total_f1']/totals['total_n']:>13.3f}")
        if totals["auto_n"]:
            print(f"{'MICRO-AUTO':<10}{totals['auto_n']:>4}  {totals['micro_auto']:>10.3f}"
                  f"  {'(official auto)':>13}")
        for label, names in (("PATHAN", sorted(PATHAN_TRACKS)),
                             ("LEKHAN", sorted(LEKHAN_TRACKS)),
                             ("REGISTER", [REGISTER_TRACK])):
            n, a = group_accuracy(per_track, names)
            if n:
                print(f"{label:<10}{n:>4}  {a:>10.3f}")
        print("\n* avg_tokenF1 is informative only (partial credit for wrong MCQ options).")
        print("saar/lekhan (+ grounded-long) are reference-only here.")
        print("Official score for those tracks = LLM-judge (1-5 rubric) + human audit.")
        print(f"Missing preds: {sum(1 for _id in gold if _id not in preds)}/{totals['total_n']}")

    if args.out:
        outp = Path(args.out)
        if outp.parent and str(outp.parent) not in ("", "."):
            outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "rows": rows,
            "per_track": {k: {"n": v["n"],
                              "acc": v["exact"] / v["n"] if v["n"] else 0,
                              "avg_f1": v["f1_sum"] / v["n"] if v["n"] else 0,
                              "auto_n": v["auto_n"], "auto_exact": v["auto_exact"],
                              "judge_needed": v["judge_needed"]}
                          for k, v in per_track.items()},
            "micro": {"n": totals["total_n"], "all": totals["micro_all"],
                      "auto_n": totals["auto_n"], "auto": totals["micro_auto"]},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
