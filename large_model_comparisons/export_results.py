import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DEFAULT_BENCHMARK_DIR, REPO_ROOT

OUT_DIR = Path(__file__).parent / "results"


def load_run(run_dir):
    run_dir = Path(run_dir)
    cfg_p, card_p = run_dir / "run_config.json", run_dir / "scorecard_summary.json"
    if not (cfg_p.exists() and card_p.exists()):
        return None
    return run_dir, json.loads(cfg_p.read_text()), json.loads(card_p.read_text())


def is_final(cfg, card):
    return cfg.get("limit") is None and not card.get("api_errors") and card.get("total_test_samples", 0) > 0


def to_result(run_dir, cfg, card):
    o = card["overall_extended_metrics"]
    d = card["deduplicated_classification_metrics"]
    cats = card["disaggregated_sub_category_metrics"]
    op = card["operational_metrics"]
    neg = cats.get("None (Deterministic)", {})
    return {
        "model_key": card["model_key"],
        "provider": cfg["provider"],
        "api_model_id": cfg["model"],
        "served_models": card.get("served_models", []),
        "run_folder": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
        "run_started": cfg.get("started"),
        "settings": {
            "prompt_style": cfg.get("prompt_style"),
            "shots": cfg.get("shots"),
            "reasoning_effort": cfg.get("effort"),
            "temperature": cfg.get("temperature"),
            "max_output_tokens": cfg.get("max_tokens"),
            "concurrent_workers": cfg.get("workers"),
            "parser": card.get("parser"),
            "test_data": str(Path(cfg["data"]).relative_to(REPO_ROOT))
            if Path(cfg["data"]).is_relative_to(REPO_ROOT) else cfg["data"],
        },
        "all_rows": {k: o[k] for k in ("n", "f1_score", "precision", "recall", "accuracy",
                                       "specificity", "balanced_accuracy", "mcc", "tp", "fp", "fn", "tn")},
        "unique_tests": {k: d[k] for k in ("n", "f1_score", "precision", "recall", "accuracy",
                                           "specificity", "balanced_accuracy", "mcc", "tp", "fp", "fn", "tn")},
        "recall_by_root_cause": {
            cat: {"recall": m["recall"], "detected": m["tp"], "n": m["tp"] + m["fn"]}
            for cat, m in cats.items() if cat != "None (Deterministic)"
        },
        "non_flaky_specificity": {
            "specificity": round(neg["tn"] / (neg["tn"] + neg["fp"]), 4) if neg else None,
            "correctly_cleared": neg.get("tn"), "n": (neg.get("tn", 0) + neg.get("fp", 0)) or None,
        },
        "root_cause_category_accuracy": card["root_cause_category_accuracy"],
        "reliability": {
            "parsing_errors": card["parsing_errors"],
            "api_errors": card.get("api_errors", 0),
            "stop_reasons": card.get("stop_reasons", {}),
        },
        "operational": {
            "api_calls": op.get("api_calls"),
            "api_input_tokens": op.get("api_input_tokens"),
            "api_output_tokens": op.get("api_output_tokens"),
            "mean_request_latency_sec": op.get("mean_request_latency_sec"),
            "wall_time_sec": op.get("total_inference_time_sec"),
            "avg_wall_time_per_unique_prompt_sec": op.get("avg_latency_per_sample_sec"),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    ap.add_argument("--run", action="append", default=[], help="use this run folder for its model")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    chosen, skipped = {}, []
    for run_dir in sorted(Path(args.benchmark_dir).glob("run_*")):
        loaded = load_run(run_dir)
        if not loaded:
            continue
        _, cfg, card = loaded
        if not is_final(cfg, card):
            skipped.append(run_dir.name)
            continue
        prev = chosen.get(card["model_key"])
        if prev is None or cfg.get("started", "") > prev[1].get("started", ""):
            chosen[card["model_key"]] = loaded
    for r in args.run:
        loaded = load_run(r)
        if not loaded:
            sys.exit(f"[-] {r} is not a commercial run folder")
        chosen[loaded[2]["model_key"]] = loaded

    if not chosen:
        sys.exit("[-] no complete full-test-split runs found")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, (run_dir, cfg, card) in sorted(chosen.items()):
        res = to_result(run_dir, cfg, card)
        results[key] = res
        (out / f"{key}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(f"[+] {out / (key + '.json')}  <- {run_dir.name}")

    summary = {
        "note": ("all_rows matches benchmark_pipeline scoring (duplicate rows counted); "
                 "unique_tests counts each test once and is ~88% flaky, so prefer "
                 "balanced_accuracy / mcc there."),
        "models": {
            k: {"f1_all_rows": r["all_rows"]["f1_score"], "accuracy_all_rows": r["all_rows"]["accuracy"],
                "mcc_all_rows": r["all_rows"]["mcc"], "f1_unique": r["unique_tests"]["f1_score"],
                "balanced_accuracy_unique": r["unique_tests"]["balanced_accuracy"],
                "mcc_unique": r["unique_tests"]["mcc"],
                "root_cause_accuracy": r["root_cause_category_accuracy"]["accuracy"],
                "recall_by_root_cause": {c: v["recall"] for c, v in r["recall_by_root_cause"].items()},
                "non_flaky_specificity": r["non_flaky_specificity"]["specificity"],
                "reasoning_effort": r["settings"]["reasoning_effort"],
                "mean_request_latency_sec": r["operational"]["mean_request_latency_sec"]}
            for k, r in results.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[+] {out / 'summary.json'}")
    if skipped:
        print(f"[*] skipped pilot/failed runs: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
