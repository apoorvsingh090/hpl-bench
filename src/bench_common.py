"""Shared harness helpers for TPL-Bench (stdlib only).

One source of truth for text normalization, tokenization, JSONL loading and
HTTP chat transport with retry/backoff. Language-specific values (strip
characters, extra replacements, prompts, tracks) are passed in by callers.
"""
import json
import random
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from functools import lru_cache

DEFAULT_STRIP = " \t\n\r'\"“”‘’.,!?;:()[]{}-–—"


def to_str(s):
    if s is None:
        return ""
    if isinstance(s, str):
        return s
    return str(s)


@lru_cache(maxsize=200_000)
def normalize(s, strip_chars=DEFAULT_STRIP, replacements=()):
    """NFC + zero-width/NBSP cleanup + script-specific swaps + casefold (ASCII)."""
    s = to_str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u200c", "").replace("\u200d", "")
    s = s.replace("\u00a0", " ")
    for old, new in replacements:
        s = s.replace(old, new)
    s = s.strip().strip(strip_chars).strip()
    s = " ".join(s.split())
    if any(("a" <= c.lower() <= "z") for c in s):
        s = s.casefold()
    return s


@lru_cache(maxsize=200_000)
def tokenize(s, strip_chars=DEFAULT_STRIP, replacements=()):
    s = normalize(s, strip_chars, replacements)
    return tuple(t for t in (t.strip(strip_chars) for t in s.split(" ")) if t)


@lru_cache(maxsize=200_000)
def token_counter(s, strip_chars=DEFAULT_STRIP, replacements=()):
    return Counter(tokenize(s, strip_chars, replacements))


def token_f1_tokens(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_tokens)
    r = overlap / len(gold_tokens)
    return 2 * p * r / (p + r)


def token_f1(pred, gold, strip_chars=DEFAULT_STRIP, replacements=()):
    return token_f1_tokens(
        tokenize(pred, strip_chars, replacements),
        tokenize(gold, strip_chars, replacements),
    )


class InputError(Exception):
    """Raised for malformed/absent input files; CLIs convert to exit code 1."""


def load_jsonl(path, what="input", require_id=True):
    """Load a JSONL file; raises InputError with a line-numbered message."""
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        raise InputError(f"{what} file not found: {path}")
    items = []
    with fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise InputError(f"{what} line {i}: invalid JSON: {e}")
            if require_id and (not isinstance(obj, dict) or "id" not in obj):
                raise InputError(f"{what} line {i}: each line must be an object with 'id'")
            items.append(obj)
    return items


def fail(message, code=1):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def post_json(url, body, headers=None, timeout=300, retries=4, log=None):
    """POST JSON with exponential backoff on transient HTTP/network errors.

    Returns the decoded JSON response. Raises HTTPError/URLError when retries
    are exhausted so callers can decide the exit path.
    """
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    last_exc = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in RETRY_STATUS and attempt < retries:
                delay = min(60.0, (2 ** attempt) + random.random())
                if log:
                    log(f"HTTP {e.code}; retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
        except urllib.error.URLError as e:
            last_exc = e
            if attempt < retries:
                delay = min(60.0, (2 ** attempt) + random.random())
                if log:
                    log(f"network error ({e.reason}); retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
    raise last_exc  # pragma: no cover - loop always returns or raises


def chat_content(data):
    """Extract choices[0].message.content from an OpenAI-compatible response."""
    try:
        return str(data["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, AttributeError):
        raise InputError(f"unexpected chat response: {str(data)[:300]}")
