"""Run HPL-Bench on Hugging Face cloud (no local GPU).

Uses Hugging Face Inference Providers (OpenAI-compatible router, stdlib only).
Compute happens on HF servers — your machine just sends prompts and scores.

Setup:
  1. Create a free account at https://huggingface.co/join
  2. Create a token at https://huggingface.co/settings/tokens (read-only is enough)
  3. $env:HF_TOKEN=\"hf_...\"  (PowerShell)  /  export HF_TOKEN=hf_...  (bash)

Usage:
  python src/run_hf_api.py --gold data/dev/hpl_dev.jsonl --out preds_hf.jsonl --model google/gemma-2-2b-it
  python src/run_hf_api.py --gold data/dev/hpl_dev.jsonl --out preds_hf.jsonl --model sarvamai/sarvam-2b-v0.5 --limit 5
  python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds preds_hf.jsonl

Notes:
  - Free tier has rate limits; use --limit 5 first, then full run.
  - Only the "prediction" key is written, compatible with src/evaluate.py.
  - Prompts are the same fixed Hindi templates as src/run_local.py.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_local import build_prompt, load_gold

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


def hf_chat(prompt: str, model: str, token: str, temperature=0.0, max_tokens=256) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        ROUTER_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code == 401:
            print("ERROR: bad HF_TOKEN (401). Check https://huggingface.co/settings/tokens",
                  file=sys.stderr)
        elif e.code == 429:
            print("ERROR: rate-limited (429). Wait and retry with --limit 5 first.",
                  file=sys.stderr)
        print(f"ERROR: HF HTTP {e.code}: {detail}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach {ROUTER_URL} ({e})", file=sys.stderr)
        raise SystemExit(1)
    try:
        return str(data["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, AttributeError):
        print(f"ERROR: unexpected HF response: {str(data)[:300]}", file=sys.stderr)
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="e.g. google/gemma-2-2b-it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls (rate-limit friendly)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: set HF_TOKEN first (https://huggingface.co/settings/tokens)", file=sys.stderr)
        raise SystemExit(2)

    items = load_gold(args.gold, args.limit)
    print(f"loaded {len(items)} items; model={args.model} (HF cloud, no local GPU)",
          file=sys.stderr)
    preds = []
    for i, it in enumerate(items, 1):
        preds.append(hf_chat(build_prompt(it), args.model, token,
                             args.temperature, args.max_tokens))
        print(f"  {i}/{len(items)} {it.get('id')}", file=sys.stderr)
        if i < len(items):
            time.sleep(args.sleep)

    with open(args.out, "w", encoding="utf-8") as f:
        for it, p in zip(items, preds):
            f.write(json.dumps({"id": it.get("id"), "prediction": p}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(items)} preds via HF cloud)")


if __name__ == "__main__":
    main()
