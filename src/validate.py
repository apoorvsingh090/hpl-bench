"""Validate HPL-Bench JSONL against schema/hpl_schema.json (stdlib only)."""
import argparse
import json
import re
import sys
from pathlib import Path

ID_RE = re.compile(r"^HPL-[A-Z]{2}-[0-9]{3}$")
TRACKS = {"padh", "anumaan", "vyakaran", "saar", "lekhan", "register", "grounded"}
LEVELS = {"A1", "A2", "B1", "B2", "C1"}
ID_PREFIX = {"padh": "PA", "anumaan": "AN", "vyakaran": "VY", "saar": "SA",
             "lekhan": "LE", "register": "RE", "grounded": "GR"}


def validate_item(obj, lineno):
    ok = True
    if not isinstance(obj, dict):
        print(f"line {lineno}: item must be a JSON object", file=sys.stderr)
        return False
    for f in ("id", "track", "level", "prompt_hi", "source", "version"):
        if f not in obj:
            print(f"line {lineno}: missing required field '{f}'", file=sys.stderr)
            ok = False
    _id = obj.get("id")
    if "id" in obj:
        if not isinstance(_id, str) or not ID_RE.match(_id):
            print(f"line {lineno}: bad id '{_id}' (want HPL-XX-000)", file=sys.stderr)
            ok = False
    track = obj.get("track")
    if track not in TRACKS:
        print(f"line {lineno}: bad track '{track}'", file=sys.stderr)
        ok = False
    elif isinstance(_id, str) and ID_RE.match(_id):
        want = ID_PREFIX.get(track)
        got = _id.split("-")[1] if "-" in _id else ""
        if want and got != want:
            print(f"line {lineno}: warning: id prefix '{got}' != track '{track}' (want {want})",
                  file=sys.stderr)
    if obj.get("level") not in LEVELS:
        print(f"line {lineno}: bad level '{obj.get('level')}'", file=sys.stderr)
        ok = False
    ph = obj.get("prompt_hi", "")
    if not isinstance(ph, str) or not ph.strip():
        print(f"line {lineno}: empty/invalid prompt_hi (must be non-empty string)", file=sys.stderr)
        ok = False
    if "choices" in obj:
        ch = obj["choices"]
        if not isinstance(ch, list) or not (2 <= len(ch) <= 5):
            print(f"line {lineno}: choices must be a list of 2-5 options", file=sys.stderr)
            ok = False
        else:
            for c in ch:
                if not isinstance(c, str) or not c.strip():
                    print(f"line {lineno}: choices must be non-empty strings", file=sys.stderr)
                    ok = False
                    break
            if "answer" in obj and isinstance(obj["answer"], str) and obj["answer"] not in ch:
                print(f"line {lineno}: answer must exactly equal one choice", file=sys.stderr)
                ok = False
    if "acceptable_answers" in obj:
        aa = obj["acceptable_answers"]
        if not isinstance(aa, list) or any(not isinstance(a, str) for a in aa):
            print(f"line {lineno}: acceptable_answers must be a list of strings", file=sys.stderr)
            ok = False
    for f in ("answer", "passage_hi", "explanation_hi", "source", "version"):
        if f in obj and obj[f] is not None and not isinstance(obj[f], str):
            print(f"line {lineno}: field '{f}' must be a string", file=sys.stderr)
            ok = False
    if track in ("lekhan", "saar") and "rubric" not in obj:
        print(f"line {lineno}: {track} items require 'rubric'", file=sys.stderr)
        ok = False
    if "rubric" in obj and not isinstance(obj["rubric"], dict):
        print(f"line {lineno}: rubric must be an object", file=sys.stderr)
        ok = False
    if track in ("padh", "saar", "grounded") and not obj.get("passage_hi"):
        print(f"line {lineno}: warning: {track} usually needs passage_hi", file=sys.stderr)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL file to validate")
    args = ap.parse_args()
    path = Path(args.data)
    seen = set()
    ok_all = True
    n = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {i}: invalid JSON: {e}", file=sys.stderr)
                ok_all = False
                continue
            _id = obj.get("id") if isinstance(obj, dict) else None
            if _id in seen:
                print(f"line {i}: duplicate id '{_id}'", file=sys.stderr)
                ok_all = False
            seen.add(_id)
            if not validate_item(obj, i):
                ok_all = False
    print(f"checked {n} items in {path}")
    if ok_all:
        print("OK: schema validation passed")
        return 0
    print("FAIL: validation errors found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
