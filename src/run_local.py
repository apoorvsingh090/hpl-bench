"""Run HPL-Bench on a local model and emit preds JSONL for src/evaluate.py.

Backends (all optional, stdlib-first):
  dummy          — no model, writes empty predictions (pipeline smoke test)
  hf             — HuggingFace transformers (pip install transformers torch --upgrade)
  ollama         — local Ollama server (stdlib urllib, no extra deps)
  openai-compat  — any OpenAI-compatible server: vLLM, LM Studio, Ollama, llama.cpp server

Predictions are appended to --out as they complete, so an interrupted run is
never lost: re-run with --resume to skip ids already present.

Examples (PowerShell):
  # 1. smoke test (no model needed)
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_dummy.jsonl --backend dummy

  # 2. Ollama (e.g. gemma3, aya-expanse, sarvam, qwen3)
  # ollama pull gemma3:4b
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_ollama.jsonl --backend ollama --model gemma3:4b

  # 3. HuggingFace (e.g. Sarvam-2B, Gemma-2-2b-it, Llama-3.2-3B-Instruct)
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_hf.jsonl --backend hf --model google/gemma-2-2b-it --batch-size 4 --device cuda

  # 4. vLLM / LM Studio (OpenAI-compatible endpoint)
  # vllm serve google/gemma-2-2b-it --port 8000
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_vllm.jsonl --backend openai-compat --model google/gemma-2-2b-it --base-url http://localhost:8000/v1 --resume

Then score:
  python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds preds_ollama.jsonl
"""
import argparse
import json
import os
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from bench_common import InputError, chat_content, fail, load_jsonl, post_json


def build_prompt(item: dict) -> str:
    passage = str(item.get("passage_hi", "") or "").strip()
    prompt = str(item.get("prompt_hi", "") or "").strip()
    track = item.get("track", "")
    choices = item.get("choices") if isinstance(item.get("choices"), list) else None

    if track == "grounded":
        # Grounded prompt always carries citation/abstention instructions,
        # with choices appended when the item is MCQ.
        base = ("निर्देश: उत्तर केवल दिए गए अंशों के आधार पर दें और स्रोत "
                "[अंश 1]/[अंश 2] उद्धृत करें। अंशों में उत्तर न हो तो लिखें: "
                "जानकारी उपलब्ध नहीं है।\n")
        if passage:
            base += f"अंश: {passage}\n"
        base += f"प्रश्न: {prompt}\n"
        if choices:
            base += "विकल्प:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices)) + "\n"
        return base + "उत्तर:"
    if choices:
        ch_txt = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
        base = "निर्देश: केवल सही उत्तर लिखें, कोई स्पष्टीकरण न दें।\n"
        if passage:
            base += f"गद्यांश: {passage}\n"
        base += f"प्रश्न: {prompt}\nविकल्प:\n{ch_txt}\nउत्तर:"
        return base
    if track == "vyakaran":
        return ("निर्देश: अशुद्ध वाक्य को शुद्ध करके केवल शुद्ध वाक्य लिखें।\n"
                f"अशुद्ध: {prompt}\nशुद्ध:")
    if track == "saar":
        return ("निर्देश: निम्नलिखित गद्यांश का सारांश लिखें। "
                "नाम, संख्या और तथ्य न बदलें, कोई नई जानकारी न जोड़ें।\n"
                f"गद्यांश: {passage or prompt}\nसारांश:")
    if track == "lekhan":
        return f"निर्देश: {prompt}\nशब्द-सीमा का पालन करें। शुद्ध, स्पष्ट और संगठित हिंदी में लिखें।\n"
    base = "निर्देश: केवल सही उत्तर लिखें, कोई स्पष्टीकरण न दें।\n"
    if passage:
        base += f"गद्यांश: {passage}\n"
    base += f"प्रश्न: {prompt}\nउत्तर:"
    return base


def load_gold(path, limit=None):
    try:
        items = load_jsonl(path, "gold")
    except InputError as e:
        fail(str(e))
    items.sort(key=lambda o: o.get("id", ""))
    if limit is None:
        return items
    if limit <= 0:
        return []
    return items[:limit]


def load_existing(out_path):
    """Ids already scored in --out (for --resume); malformed file is an error."""
    if not os.path.exists(out_path):
        return set()
    try:
        return {o["id"] for o in load_jsonl(out_path, "existing preds")}
    except InputError as e:
        fail(f"{e} (delete the file or drop --resume)")


def append_pred(fh, item_id, prediction, model, backend):
    fh.write(json.dumps({"id": item_id, "prediction": prediction,
                         "model": model, "backend": backend},
                        ensure_ascii=False) + "\n")
    fh.flush()


def _hf_device_kwargs(device):
    if device == "cuda":
        return {"device": 0}
    if device == "mps":
        return {"device": "mps"}
    if device == "auto":
        return {"device_map": "auto"}
    return {"device": -1}


def gen_hf(prompts, model, max_new_tokens=256, temperature=0.0,
           batch_size=1, device="cpu"):
    try:
        from transformers import pipeline
    except ImportError:
        fail("pip install transformers torch --upgrade")
    do_sample = temperature > 0
    kwargs = dict(model=model, trust_remote_code=True, do_sample=do_sample,
                  max_new_tokens=max_new_tokens, **_hf_device_kwargs(device))
    if do_sample:
        kwargs["temperature"] = temperature
    pipe = pipeline("text-generation", **kwargs)
    gen_kwargs = {"return_full_text": False, "batch_size": max(1, batch_size)}
    out = []
    for i in range(0, len(prompts), max(1, batch_size)):
        batch = prompts[i:i + max(1, batch_size)]
        results = pipe(batch, **gen_kwargs)
        for r in results:
            text = r[0]["generated_text"] if isinstance(r, list) else r["generated_text"]
            out.append(str(text).strip())
    return out


def ollama_generate(prompt, model, host="http://localhost:11434",
                    options=None, retries=4) -> str:
    host = (host or "").rstrip("/")
    body = {"model": model, "prompt": prompt, "stream": False, "options": options or {}}
    try:
        data = post_json(f"{host}/api/generate", body, timeout=300, retries=retries,
                         log=lambda m: print(f"  ollama: {m}", file=sys.stderr))
    except urllib.error.HTTPError as e:
        fail(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        fail(f"cannot reach Ollama at {host} ({e}). Is `ollama serve` running?")
    except InputError as e:
        fail(f"bad Ollama response: {e}")
    return str(data.get("response", "") or "").strip()


def openai_compat_generate(prompt, model, base_url, api_key="sk-no-key",
                           temperature=0.0, max_tokens=512, retries=4) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    }
    try:
        data = post_json(f"{base_url.rstrip('/')}/chat/completions", body,
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=300, retries=retries,
                         log=lambda m: print(f"  server: {m}", file=sys.stderr))
    except urllib.error.HTTPError as e:
        fail(f"server HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        fail(f"cannot reach server at {base_url} ({e})")
    try:
        return chat_content(data)
    except InputError as e:
        fail(str(e))


def run_api_backend(items, out_fh, args, generate):
    """Run per-item generation, optionally concurrently, appending as we go."""
    def work(it):
        return it, generate(build_prompt(it))

    total = len(items)
    if args.workers <= 1:
        for i, it in enumerate(items, 1):
            _, text = work(it)
            append_pred(out_fh, it.get("id"), text, args.model, args.backend)
            print(f"  {i}/{total} {it.get('id')}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, it): it for it in items}
            for i, fut in enumerate(futures, 1):
                it, text = fut.result()
                append_pred(out_fh, it.get("id"), text, args.model, args.backend)
                print(f"  {i}/{total} {it.get('id')}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="dummy",
                    choices=["dummy", "hf", "ollama", "openai-compat"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--api-key", default="sk-no-key")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="score first N items (quick smoke test)")
    ap.add_argument("--batch-size", type=int, default=1, help="HF batch size")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests for HTTP backends (default 1)")
    ap.add_argument("--retries", type=int, default=4, help="retries on 429/5xx/network errors")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out and skip ids already present")
    args = ap.parse_args()

    items = load_gold(args.gold, args.limit)
    print(f"loaded {len(items)} items from {args.gold}", file=sys.stderr)

    existing = load_existing(args.out) if args.resume else set()
    if existing:
        before = len(items)
        items = [it for it in items if it.get("id") not in existing]
        print(f"resume: {before - len(items)} items already scored, {len(items)} to go",
              file=sys.stderr)
    mode = "a" if args.resume else "w"

    if args.backend != "dummy" and not args.model:
        fail("--model required for this backend", code=2)

    with open(args.out, mode, encoding="utf-8") as out_fh:
        if args.backend == "dummy":
            for it in items:
                append_pred(out_fh, it.get("id"), "", "", "dummy")
        elif args.backend == "hf":
            prompts = [build_prompt(it) for it in items]
            texts = gen_hf(prompts, args.model, args.max_new_tokens, args.temperature,
                           args.batch_size, args.device)
            for it, text in zip(items, texts):
                append_pred(out_fh, it.get("id"), text, args.model, "hf")
        elif args.backend == "ollama":
            def generate(prompt):
                return ollama_generate(
                    prompt, args.model, args.ollama_host,
                    {"temperature": args.temperature, "num_predict": args.max_new_tokens},
                    retries=args.retries)
            run_api_backend(items, out_fh, args, generate)
        else:
            def generate(prompt):
                return openai_compat_generate(prompt, args.model, args.base_url,
                                              args.api_key, args.temperature,
                                              args.max_new_tokens, retries=args.retries)
            run_api_backend(items, out_fh, args, generate)

    print(f"wrote {args.out} (backend={args.backend})")


if __name__ == "__main__":
    main()
