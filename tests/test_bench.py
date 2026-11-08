"""Tests for the HPL-Bench harness (scorer, normalizer, validator, CLIs)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import bench_common  # noqa: E402
import evaluate  # noqa: E402

GOLD = REPO_ROOT / "data" / "dev" / "hpl_dev.jsonl"
SAMPLE_PREDS = REPO_ROOT / "examples" / "sample_preds.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, *args], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8",
    )


def test_normalize_folds_chandrabindu_into_anusvara():
    assert evaluate.normalize_text("कँ") == "कं"


def test_normalize_strips_danda_and_zwj():
    assert evaluate.normalize_text(" अ।  \u200c ") == "अ"
    assert bench_common.normalize("TAMIL ok") == "tamil ok"


def test_tokenize_and_f1():
    assert bench_common.token_f1("a b c", "a b c") == 1.0
    assert bench_common.token_f1("a b", "x y") == 0.0
    assert bench_common.token_f1("a b", "a c") == pytest.approx(0.5)
    assert bench_common.token_f1("", "") == 0.0


def test_resolve_index_variants():
    choices = ["alpha", "beta", "gamma"]
    assert evaluate.resolve_index("1", choices) == "alpha"
    assert evaluate.resolve_index("(b)", choices) == "beta"
    assert evaluate.resolve_index("2.", choices) == "beta"
    assert evaluate.resolve_index("9", choices) == "9"
    assert evaluate.resolve_index("zzz", choices) == "zzz"
    assert evaluate.resolve_index("1", None) == "1"


def test_score_item_mcq_exact_via_index():
    item = {"track": "padh", "answer": "beta", "choices": ["alpha", "beta"]}
    assert evaluate.score_item(item, "2")["exact"] == 1
    assert evaluate.score_item(item, "gamma")["exact"] == 0


def test_score_item_accepts_alternate_answers():
    item = {"track": "padh", "answer": "alpha",
            "acceptable_answers": ["अल्फ़ा"]}
    assert evaluate.score_item(item, "अल्फ़ा")["exact"] == 1


def test_score_item_empty_gold_is_skipped():
    result = evaluate.score_item({"track": "lekhan", "answer": ""}, "anything")
    assert result["skipped_auto"] is True


def test_micro_auto_is_bounded():
    gold_items = bench_common.load_jsonl(str(GOLD), "gold")
    gold = evaluate.build_gold_map(gold_items)
    preds = evaluate.build_pred_map(
        bench_common.load_jsonl(str(SAMPLE_PREDS), "preds"))
    per_track, rows = evaluate.evaluate(gold, preds)
    totals = evaluate.aggregate(per_track)
    assert 0.0 <= totals["micro_all"] <= 1.0
    assert 0.0 <= totals["micro_auto"] <= 1.0
    assert totals["auto_n"] <= totals["total_n"]
    assert rows and all("needs_judge" in r for r in rows)


def test_validate_cli_passes_seed_data():
    result = run_cli("src/validate.py", "--data", str(GOLD))
    assert result.returncode == 0, result.stderr
    assert "OK: schema validation" in result.stdout


def test_validate_cli_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "HPL-PA-001", "track": "nope"}\n', encoding="utf-8")
    result = run_cli("src/validate.py", "--data", str(bad))
    assert result.returncode == 1


def test_validate_cli_rejects_missing_file():
    result = run_cli("src/validate.py", "--data", "does-not-exist.jsonl")
    assert result.returncode == 1


def test_evaluate_cli_smoke():
    result = run_cli("src/evaluate.py", "--gold", str(GOLD), "--preds", str(SAMPLE_PREDS))
    assert result.returncode == 0, result.stderr
    assert "MICRO-ALL" in result.stdout


def test_evaluate_cli_writes_full_out_json(tmp_path):
    out = tmp_path / "eval.json"
    result = run_cli("src/evaluate.py", "--gold", str(GOLD),
                     "--preds", str(SAMPLE_PREDS), "--out", str(out))
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["micro"]["n"] >= 1
    for entry in payload["per_track"].values():
        assert {"n", "acc", "avg_f1", "auto_n", "auto_exact", "judge_needed"} <= set(entry)


def test_run_local_dummy_then_evaluate(tmp_path):
    preds = tmp_path / "preds.jsonl"
    res = run_cli("src/run_local.py", "--gold", str(GOLD), "--out", str(preds),
                  "--backend", "dummy", "--limit", "3")
    assert res.returncode == 0, res.stderr
    lines = [json.loads(line) for line in preds.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    assert len(lines) == 3
    assert all(set(o) >= {"id", "prediction"} for o in lines)

    res = run_cli("src/evaluate.py", "--gold", str(GOLD), "--preds", str(preds))
    assert res.returncode == 0, res.stderr


def test_run_local_resume_skips_existing(tmp_path):
    preds = tmp_path / "preds.jsonl"
    base = ["src/run_local.py", "--gold", str(GOLD), "--out", str(preds),
            "--backend", "dummy", "--limit", "2", "--resume"]
    assert run_cli(*base).returncode == 0
    assert run_cli(*base).returncode == 0
    lines = [line for line in preds.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    assert len(lines) == 2  # no duplicates on the second run
