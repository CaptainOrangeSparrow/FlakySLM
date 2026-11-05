import glob
import json
import os
import pandas as pd
import config


def aggregate_all_benchmarks():
    print("--- AGGREGATING BASELINE BENCHMARK RESULTS ---")
    
    benchmark_dir = config.BASE_BENCHMARK_DIR
    summary_files = glob.glob(os.path.join(benchmark_dir, "run_*", "scorecard_summary.json"))

    if not summary_files:
        print(f"[-] No benchmark summary scorecards found in '{benchmark_dir}'.")
        return

    print(f"[*] Found {len(summary_files)} scorecards across past evaluation runs.")

    summary_rows = []
    for file_path in summary_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            op_metrics = data.get("operational_metrics", {})
            overall = data.get("overall_classification_metrics", {})
            disaggregated = data.get("disaggregated_sub_category_metrics", {})

            row = {
                "Run Folder": os.path.basename(os.path.dirname(file_path)),
                "Model Key": data.get("model_key", ""),
                "Model Name": data.get("model_name", ""),
                "Run Tag": data.get("run_tag", ""),
                "Timestamp": data.get("timestamp", ""),
                "F1-Score": overall.get("f1_score", 0.0),
                "Precision": overall.get("precision", 0.0),
                "Recall": overall.get("recall", 0.0),
                "Accuracy": overall.get("accuracy", 0.0),
                "Avg Latency (s)": op_metrics.get("avg_latency_per_sample_sec", 0.0),
                "Total Tokens": op_metrics.get("total_tokens_consumed", 0),
                "Parsing Errors": data.get("parsing_errors", 0),
                # Disaggregated F1 sub-scores
                "F1 (Order-Dependent)": disaggregated.get("Order-Dependent (OD)", {}).get("f1_score", 0.0),
                "F1 (Infrastructure)": disaggregated.get("Infrastructure / Environment", {}).get("f1_score", 0.0),
                "F1 (General)": disaggregated.get("General Flakiness", {}).get("f1_score", 0.0),
            }
            summary_rows.append(row)
        except Exception as e:
            print(f"[-] Warning: Failed to parse {file_path}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(summary_rows)
    df.sort_values(by="F1-Score", ascending=False, inplace=True)

    # Export comparison reports
    csv_out = os.path.join(benchmark_dir, "baseline_comparative_summary.csv")
    json_out = os.path.join(benchmark_dir, "baseline_comparative_summary.json")

    df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n[SUCCESS] Baseline benchmark comparison saved to:")
    print(f" [+] CSV  : {csv_out}")
    print(f" [+] JSON : {json_out}\n")
    print(df[["Model Key", "F1-Score", "Precision", "Recall", "Avg Latency (s)", "F1 (Order-Dependent)", "F1 (Infrastructure)"]].to_string(index=False))


if __name__ == "__main__":
    aggregate_all_benchmarks()