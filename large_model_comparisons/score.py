import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CATEGORIES, build_scorecard, legacy_parse, parse_prediction, read_jsonl


def rescore_run(run_dir, overwrite=False):
    run_dir = Path(run_dir)
    log_path, card_path = run_dir / "predictions_log.jsonl", run_dir / "scorecard_summary.json"
    logs = read_jsonl(log_path)
    if not logs or not card_path.exists():
        return None
    old = json.loads(card_path.read_text())
    if overwrite and (run_dir / "scorecard_summary.orig.json").exists():
        old = json.loads((run_dir / "scorecard_summary.orig.json").read_text())

    for r in logs:
        pred, cat, method = parse_prediction(r.get("raw_output", ""))
        r.update(predicted_label=pred, predicted_category=cat, parse_method=method,
                 parsing_error=(method == "unparsed"),
                 predicted_label_legacy_parser=legacy_parse(r.get("raw_output", "")))

    keep = {k: old[k] for k in ("provider", "served_models", "prompt_style", "shots", "api_errors",
                                "stop_reasons") if k in old}
    card = build_scorecard(logs, old.get("model_name", ""), old.get("model_key", run_dir.name),
                           old.get("run_tag", ""), old.get("operational_metrics", {}), extra=keep)
    card["timestamp"] = old.get("timestamp", card["timestamp"])
    card["legacy_parser_overall_metrics"] = old.get("overall_classification_metrics")

    (run_dir / "scorecard_rescored.json").write_text(json.dumps(card, indent=2))
    if overwrite:
        backup = run_dir / "scorecard_summary.orig.json"
        if not backup.exists():
            shutil.copy(card_path, backup)
        card_path.write_text(json.dumps(card, indent=2))
        with log_path.open("w", encoding="utf-8") as f:
            for r in logs:
                f.write(json.dumps(r) + "\n")
    return card


COLS = [
    ("run", lambda c: f"{c['model_key']}/{c.get('run_tag', '')}"),
    ("n_rows", lambda c: c["total_test_samples"]),
    ("F1", lambda c: c["overall_classification_metrics"]["f1_score"]),
    ("P", lambda c: c["overall_classification_metrics"]["precision"]),
    ("R", lambda c: c["overall_classification_metrics"]["recall"]),
    ("Acc", lambda c: c["overall_classification_metrics"]["accuracy"]),
    ("MCC", lambda c: c["overall_extended_metrics"]["mcc"]),
    ("n_uniq", lambda c: c["deduplicated_classification_metrics"]["n"]),
    ("uniq_F1", lambda c: c["deduplicated_classification_metrics"]["f1_score"]),
    ("uniq_BalAcc", lambda c: c["deduplicated_classification_metrics"]["balanced_accuracy"]),
    ("uniq_MCC", lambda c: c["deduplicated_classification_metrics"]["mcc"]),
    ("RootCauseAcc", lambda c: c["root_cause_category_accuracy"]["accuracy"]),
    ("ParseErr", lambda c: c["parsing_errors"]),
    ("legacy_F1", lambda c: (c.get("legacy_parser_overall_metrics") or {}).get("f1_score", "")),
    ("s/sample", lambda c: c["operational_metrics"].get("avg_latency_per_sample_sec", "")),
    ("tokens", lambda c: c["operational_metrics"].get("total_tokens_consumed", "")),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run_* folders")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cards = [c for c in (rescore_run(r, args.overwrite) for r in args.runs if Path(r).is_dir()) if c]
    if not cards:
        sys.exit("[-] no run folders with predictions_log.jsonl + scorecard_summary.json")
    cards.sort(key=lambda c: -c["overall_classification_metrics"]["f1_score"])

    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    print("| " + " | ".join(name for name, _ in COLS) + " |")
    print("|" + "---|" * len(COLS))
    for c in cards:
        print("| " + " | ".join(fmt(f(c)) for _, f in COLS) + " |")

    print("\nPer gold sub-category (rows): recall = share of flaky tests detected; "
          "None (Deterministic) = specificity")
    for c in cards:
        d = c["disaggregated_sub_category_metrics"]
        parts = []
        for cat in CATEGORIES:
            m = d.get(cat)
            if m:
                parts.append(f"{cat}: R={m['recall']:.3f} (n={m['tp'] + m['fn']})")
        neg = d.get("None (Deterministic)")
        if neg:
            parts.append(f"None: spec={neg['tn'] / max(1, neg['tn'] + neg['fp']):.3f} (n={neg['tn'] + neg['fp']})")
        print(f"  {c['model_key']}: " + "; ".join(parts))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([name for name, _ in COLS])
            for c in cards:
                w.writerow([f(c) for _, f in COLS])
        print(f"\n[+] wrote {args.csv}")


if __name__ == "__main__":
    main()
