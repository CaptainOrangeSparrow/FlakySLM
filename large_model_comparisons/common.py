import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_PATH = REPO_ROOT / "empirical_test_dataset" / "dataset_test.jsonl"
DEFAULT_TRAIN_PATH = REPO_ROOT / "empirical_test_dataset" / "dataset_train.jsonl"
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "benchmark_pipeline" / "benchmarks"

CATEGORIES = [
    "Order-Dependent (OD)",
    "Infrastructure / Environment",
    "General Flakiness",
]
NON_FLAKY_SUB_CATEGORY = "None (Deterministic)"

PARSER_VERSION = "classification_line_v1"


FORMAT_INSTRUCTIONS = f"""
Respond in exactly this format and nothing else:

Classification: <Flaky or Non-Flaky>
Root Cause Category: <one of: {", ".join(CATEGORIES)}, None>
Reasoning: <one or two sentences>

Category definitions:
- Order-Dependent (OD): passes or fails depending on which tests ran before it (shared state, globals, files, caches, DB rows).
- Infrastructure / Environment: depends on network, filesystem, OS, time zone, installed packages, hardware, or other external resources.
- General Flakiness: non-deterministic for other reasons (randomness, timing, concurrency, floating point, unordered collections, etc.).
Use "None" if and only if the classification is Non-Flaky.
""".strip()


def load_env_file(path=REPO_ROOT / ".env"):
    import os
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_sub_category(rec):
    if rec.get("label", "non_flaky").lower() == "non_flaky":
        return NON_FLAKY_SUB_CATEGORY
    return rec.get("metadata", {}).get("sub_category", "General Flakiness")


def load_rows(path, limit=None, seed=0):
    rows = []
    for idx, rec in enumerate(read_jsonl(path)):
        rec["_sample_index"] = idx
        rows.append(rec)
    if limit and limit < len(rows):
        rng = random.Random(seed)
        flaky = [r for r in rows if r["label"] == "flaky"]
        clean = [r for r in rows if r["label"] != "flaky"]
        n_clean = min(len(clean), limit // 2)
        n_flaky = min(len(flaky), limit - n_clean)
        rows = rng.sample(flaky, n_flaky) + rng.sample(clean, n_clean)
        rows.sort(key=lambda r: r["_sample_index"])
    return rows


def load_few_shot(train_path, k, seed=0, max_chars=1500):
    if k <= 0:
        return []
    rows = [r for r in read_jsonl(train_path) if len(r["messages"][1]["content"]) <= max_chars]
    rng = random.Random(seed)
    shots = (rng.sample([r for r in rows if r["label"] == "flaky"], (k + 1) // 2)
             + rng.sample([r for r in rows if r["label"] == "non_flaky"], k // 2))
    rng.shuffle(shots)
    msgs = []
    for r in shots:
        msgs.append({"role": "user", "content": r["messages"][1]["content"]})
        msgs.append({"role": "assistant", "content": r["messages"][2]["content"]})
    return msgs


PROMPT_STYLES = ("formatted", "dataset")


def build_messages(rec, style="formatted", few_shot=None):
    system = next(m["content"] for m in rec["messages"] if m["role"] == "system")
    user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
    if style == "formatted":
        system = f"{system}\n\n{FORMAT_INSTRUCTIONS}"
    elif style != "dataset":
        raise ValueError(f"unknown prompt style {style!r}")
    return system, list(few_shot or []) + [{"role": "user", "content": user}]


def prompt_key(system, messages):
    blob = json.dumps({"s": system, "m": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


_CLS_RE = re.compile(r"classification\**\s*[:\-]\s*\**\s*(non[\s\-_]?flaky|not\s+flaky|flaky)", re.I)
_CAT_RE = re.compile(r"root\s*cause\s*category\**\s*[:\-]\s*\**\s*([^\n*]+)", re.I)
_NEG_RE = re.compile(r"\b(non[\s\-_]?flaky|not\s+flaky)\b", re.I)
_POS_RE = re.compile(r"(?<![\w\-])flaky\b", re.I)


def _normalize_category(text):
    t = text.lower()
    if "order" in t or re.search(r"\bod\b", t):
        return "Order-Dependent (OD)"
    if "infra" in t or "environment" in t:
        return "Infrastructure / Environment"
    if "general" in t:
        return "General Flakiness"
    if "none" in t or "n/a" in t:
        return "None"
    return "Unparsed"


def parse_prediction(text):
    if not text or not text.strip():
        return "non_flaky", "Unparsed", "unparsed"
    m = _CLS_RE.search(text)
    if m:
        label = "flaky" if m.group(1).lower() == "flaky" else "non_flaky"
        c = _CAT_RE.search(text)
        return label, (_normalize_category(c.group(1)) if c else "Unparsed"), "classification_line"
    if _NEG_RE.search(text):
        return "non_flaky", "Unparsed", "keyword_fallback"
    if _POS_RE.search(text):
        return "flaky", "Unparsed", "keyword_fallback"
    return "non_flaky", "Unparsed", "unparsed"


_LEGACY_NEG = [r"\bnon-flaky\b", r"\bnon_flaky\b", r"\bnot flaky\b", r"\bdeterministic\b",
               r"\bclassification:\s*non-flaky\b"]
_LEGACY_POS = [r"\bflaky\b", r"\border-dependent\b", r"\binfrastructure\b", r"\bclassification:\s*flaky\b"]


def legacy_parse(text):
    if not text or not text.strip():
        return "non_flaky"
    t = text.lower()
    if any(re.search(p, t) for p in _LEGACY_NEG):
        return "non_flaky"
    if any(re.search(p, t) for p in _LEGACY_POS):
        return "flaky"
    return "non_flaky"


def compute_classification_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _extended_metrics(tp, fp, fn, tn):
    m = compute_classification_metrics(tp, fp, fn, tn)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m["specificity"] = round(spec, 4)
    m["balanced_accuracy"] = round((rec + spec) / 2, 4)
    m["mcc"] = round((tp * tn - fp * fn) / den, 4) if den else 0.0
    m["n"] = tp + fp + fn + tn
    return m


def _counts(logs):
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in logs:
        g, p = r["ground_truth"], r["predicted_label"]
        key = ("t" if g == p else "f") + ("p" if p == "flaky" else "n")
        c[key] += 1
    return c


def build_scorecard(logs, model_name, model_key, run_tag, operational, extra=None):
    import time

    overall = _counts(logs)
    per_cat = defaultdict(list)
    for r in logs:
        per_cat[r["sub_category"]].append(r)

    seen, dedup = set(), []
    for r in logs:
        k = (r["project_url"], r["test_identifier"], r["ground_truth"])
        if k not in seen:
            seen.add(k)
            dedup.append(r)

    both_flaky = [r for r in logs if r["ground_truth"] == "flaky" and r["predicted_label"] == "flaky"]
    cat_hits = sum(r.get("predicted_category") == r["sub_category"] for r in both_flaky)

    card = {
        "model_name": model_name,
        "model_key": model_key,
        "run_tag": run_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(logs),
        "parsing_errors": sum(bool(r["parsing_error"]) for r in logs),
        "operational_metrics": operational,
        "overall_classification_metrics": compute_classification_metrics(**overall),
        "disaggregated_sub_category_metrics": {
            cat: compute_classification_metrics(**_counts(rs)) for cat, rs in per_cat.items()
        },
        "parser": PARSER_VERSION,
        "overall_extended_metrics": _extended_metrics(**overall),
        "deduplicated_classification_metrics": _extended_metrics(**_counts(dedup)),
        "root_cause_category_accuracy": {
            "accuracy": round(cat_hits / len(both_flaky), 4) if both_flaky else 0.0,
            "n_true_positives": len(both_flaky),
        },
        "parse_methods": dict(sorted(
            ((m, sum(r.get("parse_method") == m for r in logs))
             for m in {r.get("parse_method") for r in logs}), key=lambda kv: str(kv[0]))),
    }
    if extra:
        card.update(extra)
    return card
