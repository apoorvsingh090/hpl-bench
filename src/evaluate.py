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
import unicodedata
from collections import defaultdict
from pathlib import Path

STRIP_CHARS = " \t\n\r'\"“”‘’।.,!?;:()[]{}-–—"
INDEX_RE = re.compile(r"^\(?\s*([1-5a-eA-E])\s*[.\)]?\s*$")

# Reading vs writing aggregates reported alongside per-track scores.
PATHAN_TRACKS = {"padh", "anumaan", "grounded"}
LEKHAN_TRACKS = {"vyakaran", "saar", "lekhan"}


def to_str(s) -> str:
    if s is None:
        return ""
    if isinstance(s, str):
        return s
    return str(s)


def normalize(s) -> str:
    s = to_str(s)
    s = unicodedata.normalize("NFC", s)
    # Hindi-insensitive: chandrabindu ~ anusvara, drop ZWJ/ZWNJ, NBSP -> space
    s = s.replace("\u0901", "\u0902").replace("\u200c", "").replace("\u200d", "")
    s = s.replace("\u00a0", " ")
    s = s.strip().strip(STRIP_CHARS).strip()
    s = re.sub(r"\s+", " ", s)
    if re.search(r"[A-Za-z]", s):
        s = s.casefold()
    return s


def tokenize(s):
    s = normalize(s)
    return [t for t in (t.strip(STRIP_CHARS) for t in s.split(" ")) if t]


def token_f1(pred, gold) -> float:
    from collections import Counter
    pt, gt = tokenize(pred), tokenize(gold)
    if not pt and not gt:
        return 0.0
    if not pt or not gt:
        return 0.0
    overlap = sum((Counter(pt) & Counter(gt)).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pt)
    r = overlap / len(gt)
    return 2 * p * r / (p + r)


def resolve_index(pred: str, choices) -> str:
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


def is_long_form(item) -> bool:
    if item.get("track") in ("saar", "lekhan"):
        return True
    ans = item.get("answer", "")
    if item.get("track") == "grounded" and len(tokenize(ans)) > 25:
        return True
    return False


def score_item(item, pred):
    pred = to_str(pred)
    gold = to_str(item.get("answer", ""))
    if not gold.strip():
        return {"exact": 0, "token_f1": 0.0, "skipped_auto": True}
    aa = item.get("acceptable_answers", [])
    if not isinstance(aa, list):
        aa = []
    aa = [a for a in [gold] + aa if isinstance(a, str)]
    choices = item.get("choices") if isinstance(item.get("choices"), list) else None
    candidates = [pred, resolve_index(pred, choices)]
    exact = int(any(normalize(c) == normalize(a) for c in candidates for a in aa))
    f1 = round(max([token_f1(pred, a) for a in aa] + [0.0]), 4)
    return {"exact": exact, "token_f1": f1, "skipped_auto": False}


def load_jsonl(path, what):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: {what} line {i}: invalid JSON: {e}")
                raise SystemExit(1)
            if not isinstance(o, dict) or "id" not in o:
                print(f"ERROR: {what} line {i}: each line must be an object with 'id'")
                raise SystemExit(1)
            items.append(o)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    gold_list = load_jsonl(args.gold, "gold")
    gold = {}
    for o in gold_list:
        if o["id"] in gold:
            print(f"ERROR: duplicate gold id '{o['id']}'")
            raise SystemExit(1)
        gold[o["id"]] = o

    preds = {}
    for o in load_jsonl(args.preds, "preds"):
        if o["id"] in preds:
            print(f"ERROR: duplicate pred id '{o['id']}'")
            raise SystemExit(1)
        # NOTE: only "prediction" counts; a stray "answer" key is ignored
        # so that passing the gold file as preds cannot silently ace the run.
        preds[o["id"]] = o.get("prediction", "")

    per_track = defaultdict(lambda: {"n": 0, "exact": 0, "f1_sum": 0.0,
                                     "auto_n": 0, "auto_exact": 0, "judge_needed": 0})
    rows = []
    for _id in sorted(gold):
        item = gold[_id]
        if not isinstance(item, dict) or "track" not in item:
            print(f"ERROR: gold id '{_id}' missing 'track'")
            raise SystemExit(1)
        s = score_item(item, preds.get(_id, ""))
        tr = item["track"]
        d = per_track[tr]
        d["n"] += 1
        d["exact"] += s["exact"]
        d["f1_sum"] += s["token_f1"]
        needs_judge = is_long_form(item)
        d["judge_needed"] += int(needs_judge)
        if not needs_judge and not s.get("skipped_auto"):
            d["auto_n"] += 1
            d["auto_exact"] += s["exact"]
        rows.append({"id": _id, "track": tr, **s,
                     "needs_judge": needs_judge,
                     "missing_pred": _id not in preds})

    print(f"{'track':<10}{'n':>4}  {'exact_acc':>10}  {'avg_tokenF1*':>13}  judge_needed")
    total_n = total_exact = 0
    total_f1 = 0.0
    auto_n = auto_exact = 0
    for tr in sorted(per_track):
        d = per_track[tr]
        acc = d["exact"] / d["n"] if d["n"] else 0
        avgf1 = d["f1_sum"] / d["n"] if d["n"] else 0
        print(f"{tr:<10}{d['n']:>4}  {acc:>10.3f}  {avgf1:>13.3f}  {d['judge_needed']}")
        total_n += d["n"]
        total_exact += d["exact"]
        total_f1 += d["f1_sum"]
        auto_n += d["auto_n"]
        auto_exact += d["auto_exact"]

    def agg(names):
        n = sum(per_track[t]["n"] for t in names if t in per_track)
        e = sum(per_track[t]["exact"] for t in names if t in per_track)
        return n, (e / n if n else 0)

    if total_n:
        print("-" * 55)
        print(f"{'MICRO-ALL':<10}{total_n:>4}  {total_exact/total_n:>10.3f}  {total_f1/total_n:>13.3f}")
        if auto_n:
            print(f"{'MICRO-AUTO':<10}{auto_n:>4}  {auto_exact/auto_n:>10.3f}  {'(official auto)':>13}")
        for label, names in (("PATHAN", sorted(PATHAN_TRACKS)),
                             ("LEKHAN", sorted(LEKHAN_TRACKS)),
                             ("REGISTER", ["register"])):
            n, a = agg(names)
            if n:
                print(f"{label:<10}{n:>4}  {a:>10.3f}")
        print("\n* avg_tokenF1 is informative only (partial credit for wrong MCQ options).")
        print("saar/lekhan (+ grounded-long) are reference-only here.")
        print("Official score for those tracks = LLM-judge (1-5 rubric) + human audit.")
        print(f"Missing preds: {sum(1 for _id in gold if _id not in preds)}/{total_n}")

    if args.out:
        outp = Path(args.out)
        if outp.parent and str(outp.parent) not in ("", "."):
            outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"rows": rows,
            "per_track": {k: {"n": v["n"], "acc": v["exact"]/v["n"] if v["n"] else 0,
            "avg_f1": v["f1_sum"]/v["n"] if v["n"] else 0,
            "auto_n": v["auto_n"]} for k, v in per_track.items()}},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
