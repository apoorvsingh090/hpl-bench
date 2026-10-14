"""Validate HPL-Bench JSONL against schema/hpl_schema.json (stdlib only)."""
import argparse
import json
import re
import sys
from pathlib import Path

ID_RE = re.compile(r"^HPL-[A-Z]{2}-[0-9]{3}$")
TRACKS = {"padh", "anumaan", "vyakaran", "saar", "lekhan", "register", "grounded"}
LEVELS = {"A1", "A2", "B1", "B2", "C1"}

def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    return False

def validate_item(obj, lineno):
    ok = True
    for f in ("id", "track", "level", "prompt_hi", "source", "version"):
        if f not in obj:
            print(f"line {lineno}: missing required field '{f}'", file=sys.stderr)
            ok = False
    if "id" in obj and not ID_RE.match(obj["id"]):
        print(f"line {lineno}: bad id '{obj['id']}' (want HPL-XX-000)", file=sys.stderr)
        ok = False
    if obj.get("track") not in TRACKS:
        print(f"line {lineno}: bad track '{obj.get('track')}'", file=sys.stderr)
        ok = False
    if obj.get("level") not in LEVELS:
        print(f"line {lineno}: bad level '{obj.get('level')}'", file=sys.stderr)
        ok = False
    if not obj.get("prompt_hi", "").strip():
        print(f"line {lineno}: empty prompt_hi", file=sys.stderr)
        ok = False
    if "choices" in obj:
        ch = obj["choices"]
        if not (2 <= len(ch) <= 5):
            print(f"line {lineno}: choices must have 2-5 options", file=sys.stderr)
            ok = False
        if "answer" in obj and obj["answer"] not in ch:
            print(f"line {lineno}: answer must exactly equal one choice", file=sys.stderr)
            ok = False
    if obj.get("track") == "lekhan" and "rubric" not in obj:
        print(f"line {lineno}: lekhan items require 'rubric'", file=sys.stderr)
        ok = False
    if obj.get("track") in ("padh", "grounded") and not obj.get("passage_hi"):
        print(f"line {lineno}: warning: {obj.get('track')} usually needs passage_hi", file=sys.stderr)
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
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {i}: invalid JSON: {e}", file=sys.stderr)
                ok_all = False
                continue
            if obj.get("id") in seen:
                print(f"line {i}: duplicate id '{obj.get('id')}'", file=sys.stderr)
                ok_all = False
            seen.add(obj.get("id"))
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
