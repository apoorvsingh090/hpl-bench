"""Run HPL-Bench on a local model and emit preds JSONL for src/evaluate.py.

Backends (all optional, stdlib-first):
  dummy          — no model, writes empty predictions (pipeline smoke test)
  hf             — HuggingFace transformers (pip install transformers torch --upgrade)
  ollama         — local Ollama server (stdlib urllib, no extra deps)
  openai-compat  — any OpenAI-compatible server: vLLM, LM Studio, Ollama, llama.cpp server

Examples (PowerShell):
  # 1. smoke test (no model needed)
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_dummy.jsonl --backend dummy

  # 2. Ollama (e.g. gemma3, aya-expanse, sarvam, qwen3)
  # ollama pull gemma3:4b
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_ollama.jsonl --backend ollama --model gemma3:4b

  # 3. HuggingFace (e.g. Sarvam-2B, Gemma-2-2b-it, Llama-3.2-3B-Instruct)
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_hf.jsonl --backend hf --model google/gemma-2-2b-it --max-new-tokens 256

  # 4. vLLM / LM Studio (OpenAI-compatible endpoint)
  # vllm serve google/gemma-2-2b-it --port 8000
  python src/run_local.py --gold data/dev/hpl_dev.jsonl --out preds_vllm.jsonl --backend openai-compat --model google/gemma-2-2b-it --base-url http://localhost:8000/v1

Then score:
  python src/evaluate.py --gold data/dev/hpl_dev.jsonl --preds preds_ollama.jsonl
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


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
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items.sort(key=lambda o: o.get("id", ""))
    if limit is None:
        return items
    if limit <= 0:
        return []
    return items[:limit]


# ---------------- backends ----------------

def gen_hf(prompts, model, max_new_tokens=256, temperature=0.0):
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: pip install transformers torch --upgrade", file=sys.stderr)
        raise SystemExit(1)
    do_sample = temperature > 0
    pipe = pipeline(
        "text-generation", model=model, trust_remote_code=True,
        do_sample=do_sample, temperature=temperature or None,
        max_new_tokens=max_new_tokens,
    )
    out = []
    for p in prompts:
        r = pipe(p, return_full_text=False)[0]["generated_text"]
        out.append(str(r).strip())
    return out


def ollama_generate(prompt: str, model: str, host="http://localhost:11434", options=None) -> str:
    host = (host or "").rstrip("/")
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": options or {},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return str(json.loads(r.read().decode("utf-8")).get("response", "") or "").strip()
    except urllib.error.HTTPError as e:
        print(f"ERROR: Ollama HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach Ollama at {host} ({e}). Is `ollama serve` running?", file=sys.stderr)
        raise SystemExit(1)
    except (ValueError, KeyError) as e:
        print(f"ERROR: bad Ollama response: {e}", file=sys.stderr)
        raise SystemExit(1)


def openai_compat_generate(prompt: str, model: str, base_url: str, api_key="sk-no-key", temperature=0.0, max_tokens=512) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"ERROR: server HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach server at {base_url} ({e})", file=sys.stderr)
        raise SystemExit(1)
    try:
        content = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, AttributeError) as e:
        print(f"ERROR: unexpected server response shape: {e}; body={str(data)[:300]}", file=sys.stderr)
        raise SystemExit(1)
    return str(content).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="dummy", choices=["dummy", "hf", "ollama", "openai-compat"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--api-key", default="sk-no-key")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="score first N items (quick smoke test)")
    args = ap.parse_args()

    items = load_gold(args.gold, args.limit)
    print(f"loaded {len(items)} items from {args.gold}", file=sys.stderr)

    if args.backend == "dummy":
        preds = ["" for _ in items]
    elif args.backend == "hf":
        if not args.model:
            print("ERROR: --model required for hf backend", file=sys.stderr)
            raise SystemExit(2)
        preds = gen_hf([build_prompt(it) for it in items], args.model,
                       args.max_new_tokens, args.temperature)
    elif args.backend == "ollama":
        if not args.model:
            print("ERROR: --model required, e.g. --model gemma3:4b", file=sys.stderr)
            raise SystemExit(2)
        preds = [ollama_generate(build_prompt(it), args.model, args.ollama_host,
                                 {"temperature": args.temperature,
                                  "num_predict": args.max_new_tokens}) for it in items]
    else:  # openai-compat
        if not args.model:
            print("ERROR: --model required", file=sys.stderr)
            raise SystemExit(2)
        preds = [openai_compat_generate(build_prompt(it), args.model, args.base_url,
                                        args.api_key, args.temperature, args.max_new_tokens)
                 for it in items]

    with open(args.out, "w", encoding="utf-8") as f:
        for it, p in zip(items, preds):
            f.write(json.dumps({"id": it.get("id"), "prediction": p}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(items)} preds, backend={args.backend})")


if __name__ == "__main__":
    main()
