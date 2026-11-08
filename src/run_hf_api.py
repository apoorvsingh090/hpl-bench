"""Run TPL-Bench on Hugging Face cloud (no local GPU).

Uses Hugging Face Inference Providers (OpenAI-compatible router, stdlib only).
Predictions are appended to --out as they complete: an interrupted run can be
resumed with --resume instead of being lost.

Setup:
  1. Create a free account at https://huggingface.co/join
  2. Create a token at https://huggingface.co/settings/tokens
  3. $env:HF_TOKEN="hf_..."  (PowerShell)  /  export HF_TOKEN=hf_...  (bash)

Usage:
  python src/run_hf_api.py --gold data/dev/tpl_dev.jsonl --out preds_hf.jsonl --model google/gemma-2-2b-it
  python src/run_hf_api.py --gold data/dev/tpl_dev.jsonl --out preds_hf.jsonl --model google/gemma-2-2b-it --resume
  python src/evaluate.py --gold data/dev/tpl_dev.jsonl --preds preds_hf.jsonl
"""
import argparse
import os
import sys
import time
import urllib.error

from bench_common import InputError, chat_content, fail, post_json
from run_local import append_pred, build_prompt, load_existing, load_gold

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


def hf_chat(prompt: str, model: str, token: str, temperature=0.0,
            max_tokens=256, retries=4) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        data = post_json(ROUTER_URL, body,
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=120, retries=retries,
                         log=lambda m: print(f"  hf: {m}", file=sys.stderr))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code == 401:
            fail("bad HF_TOKEN (401). Check https://huggingface.co/settings/tokens")
        if e.code == 429:
            fail("rate-limited (429) after retries. Wait, or retry with --limit 5 first.")
        fail(f"HF HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        fail(f"cannot reach {ROUTER_URL} ({e})")
    try:
        return chat_content(data)
    except InputError as e:
        fail(f"unexpected HF response: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="e.g. google/gemma-2-2b-it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="minimum seconds between call starts (rate-limit friendly)")
    ap.add_argument("--retries", type=int, default=4,
                    help="retries on 429/5xx/network errors")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out and skip ids already present")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        fail("set HF_TOKEN first (https://huggingface.co/settings/tokens)", code=2)

    items = load_gold(args.gold, args.limit)
    existing = load_existing(args.out) if args.resume else set()
    if existing:
        before = len(items)
        items = [it for it in items if it.get("id") not in existing]
        print(f"resume: {before - len(items)} items already scored, {len(items)} to go",
              file=sys.stderr)
    print(f"loaded {len(items)} items; model={args.model} (HF cloud, no local GPU)",
          file=sys.stderr)

    mode = "a" if args.resume else "w"
    last_start = 0.0
    with open(args.out, mode, encoding="utf-8") as out_fh:
        for i, it in enumerate(items, 1):
            wait = args.sleep - (time.monotonic() - last_start)
            if wait > 0:
                time.sleep(wait)
            last_start = time.monotonic()
            text = hf_chat(build_prompt(it), args.model, token,
                           args.temperature, args.max_tokens, retries=args.retries)
            append_pred(out_fh, it.get("id"), text, args.model, "hf-api")
            print(f"  {i}/{len(items)} {it.get('id')}", file=sys.stderr)

    print(f"wrote {args.out} (backend=hf-api)")


if __name__ == "__main__":
    main()
