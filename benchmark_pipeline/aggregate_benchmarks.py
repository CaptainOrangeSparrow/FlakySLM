import glob
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import config_llamacpp as config
# import config


def generate_all_visualizations(df: pd.DataFrame, aggregated_out_dir: str):
    """Generates and saves all requested publication-ready comparative visual charts."""
    if df.empty:
        return

    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # 1. Grouped Bar Chart (Per-Category F1-Scores across Models)
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    categories = ["Det F1", "OD F1", "Infra F1", "General F1"]
    cat_rename = {
        "Det F1": "Deterministic",
        "OD F1": "Order-Dependent",
        "Infra F1": "Infrastructure",
        "General F1": "General Flakiness"
    }
    
    plot_data = []
    for _, row in df.iterrows():
        for cat in categories:
            plot_data.append({
                "Model": row["Model Key"],
                "Category": cat_rename[cat],
                "F1-Score": row[cat]
            })
    
    melted_df = pd.DataFrame(plot_data)
    sns.barplot(data=melted_df, x="Category", y="F1-Score", hue="Model", palette="muted")
    plt.title("Baseline Model Performance Across Flakiness Sub-Categories", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Test Sub-Category", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Model Key", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    
    bar_path = os.path.join(aggregated_out_dir, "per_category_f1_comparison.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f" [+] Saved Visual Plot : {bar_path}")

    # -------------------------------------------------------------
    # 2. Raw Count Confusion Matrix Heatmaps
    # -------------------------------------------------------------
    num_models = len(df)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4), squeeze=False)

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[0, idx]
        cm_matrix = np.array([
            [row["Overall TN"], row["Overall FP"]],
            [row["Overall FN"], row["Overall TP"]]
        ])
        sns.heatmap(
            cm_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred: Det", "Pred: Flaky"],
            yticklabels=["Actual: Det", "Actual: Flaky"],
            ax=ax, annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(f"{row['Model Key']}\n(F1: {row['Overall F1']:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Ground Truth Label", fontsize=10)

    plt.suptitle("Raw Count Confusion Matrices (TN, FP, FN, TP)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    raw_cm_path = os.path.join(aggregated_out_dir, "raw_confusion_matrix_heatmaps.png")
    plt.savefig(raw_cm_path, dpi=300)
    plt.close()
    print(f" [+] Saved Visual Plot : {raw_cm_path}")

    # -------------------------------------------------------------
    # 3. Normalized Confusion Matrix Heatmaps (Row-normalized %)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4), squeeze=False)

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[0, idx]
        raw_cm = np.array([
            [row["Overall TN"], row["Overall FP"]],
            [row["Overall FN"], row["Overall TP"]]
        ], dtype=float)
        
        row_sums = raw_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Prevent division by zero
        norm_cm = raw_cm / row_sums

        sns.heatmap(
            norm_cm, annot=True, fmt=".2f", cmap="Purples", cbar=False, vmin=0.0, vmax=1.0,
            xticklabels=["Pred: Det", "Pred: Flaky"],
            yticklabels=["Actual: Det", "Actual: Flaky"],
            ax=ax, annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(f"{row['Model Key']}\n(Normalized Proportions)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Ground Truth Label", fontsize=10)

    plt.suptitle("Normalized Confusion Matrix Heatmaps (Proportions)", fontsize=14, fontweight="bold", y=0.95)
    plt.tight_layout()
    norm_cm_path = os.path.join(aggregated_out_dir, "normalized_confusion_matrix_heatmaps.png")
    plt.savefig(norm_cm_path, dpi=300)
    plt.close()
    print(f" [+] Saved Visual Plot : {norm_cm_path}")

    # -------------------------------------------------------------
    # 4. Radar / Spider Chart (Multi-Dimensional Capability)
    # -------------------------------------------------------------
    labels = ["Deterministic F1", "Order-Dependent F1", "Infrastructure F1", "General F1", "Precision", "Recall"]
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in df.iterrows():
        stats = [
            row["Det F1"],
            row["OD F1"],
            row["Infra F1"],
            row["General F1"],
            row["Overall Precision"],
            row["Overall Recall"]
        ]
        stats.append(stats[0])  # Close the loop for data values
        plot_angles = angles + [angles[0]]  # Close the loop for angles
        
        model_name = row["Model Key"]
        ax.plot(plot_angles, stats, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(plot_angles, stats, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    plt.title("Multi-Dimensional Model Capability Radar", fontsize=14, fontweight="bold", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    radar_path = os.path.join(aggregated_out_dir, "model_radar_capability.png")
    plt.savefig(radar_path, dpi=300)
    plt.close()
    print(f" [+] Saved Visual Plot : {radar_path}")

    # -------------------------------------------------------------
    # 5. Scatter Plot: Inference Latency vs. Overall F1-Score
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="Avg Latency (s)", y="Overall F1", hue="Model Key",
        s=150, palette="deep", style="Model Key"
    )
    
    for _, row in df.iterrows():
        plt.text(
            row["Avg Latency (s)"] + 0.0005, row["Overall F1"],
            f" {row['Model Key']}", fontsize=10, fontweight="bold", verticalalignment='center'
        )

    plt.title("Inference Latency vs. Overall F1-Score Trade-off", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Average Latency per Sample (seconds)", fontsize=12)
    plt.ylabel("Overall F1-Score", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    scatter_path = os.path.join(aggregated_out_dir, "latency_vs_f1_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f" [+] Saved Visual Plot : {scatter_path}")


def aggregate_all_benchmarks():
    print("=====================================================")
    print("   AGGREGATING BASELINE BENCHMARK RESULTS & PLOTS    ")
    print("=====================================================")
    
    benchmark_dir = config.BASE_BENCHMARK_DIR
    summary_files = glob.glob(os.path.join(benchmark_dir, "run_*", "scorecard_summary.json"))

    if not summary_files:
        print(f"[-] No benchmark summary scorecards found in '{benchmark_dir}'.")
        return

    print(f"[*] Found {len(summary_files)} scorecards across evaluation runs.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_out_dir = os.path.join(benchmark_dir, "aggregated_runs", f"agg_{timestamp}")
    os.makedirs(aggregated_out_dir, exist_ok=True)

    summary_rows = []
    
    for file_path in summary_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            op_metrics = data.get("operational_metrics", {})
            overall = data.get("overall_classification_metrics", {})
            disaggregated = data.get("disaggregated_sub_category_metrics", {})

            def get_cat_metrics(cat_key: str):
                cat = disaggregated.get(cat_key, {})
                return {
                    "P": cat.get("precision", 0.0),
                    "R": cat.get("recall", 0.0),
                    "F1": cat.get("f1_score", 0.0),
                    "TP": cat.get("tp", 0),
                    "FP": cat.get("fp", 0),
                    "FN": cat.get("fn", 0),
                    "TN": cat.get("tn", 0),
                }

            od = get_cat_metrics("Order-Dependent (OD)")
            infra = get_cat_metrics("Infrastructure / Environment")
            gen = get_cat_metrics("General Flakiness")
            det = get_cat_metrics("None (Deterministic)")

            row = {
                "Run Folder": os.path.basename(os.path.dirname(file_path)),
                "Model Key": data.get("model_key", ""),
                "Model Name": data.get("model_name", ""),
                "Run Tag": data.get("run_tag", ""),
                "Timestamp": data.get("timestamp", ""),
                
                "Overall F1": overall.get("f1_score", 0.0),
                "Overall Precision": overall.get("precision", 0.0),
                "Overall Recall": overall.get("recall", 0.0),
                "Overall Accuracy": overall.get("accuracy", 0.0),
                
                "Overall TP": overall.get("tp", 0),
                "Overall FP": overall.get("fp", 0),
                "Overall FN": overall.get("fn", 0),
                "Overall TN": overall.get("tn", 0),

                "Avg Latency (s)": op_metrics.get("avg_latency_per_sample_sec", 0.0),
                "Total Tokens": op_metrics.get("total_tokens_consumed", 0),
                "Parsing Errors": data.get("parsing_errors", 0),
                
                "OD Precision": od["P"], "OD Recall": od["R"], "OD F1": od["F1"],
                "OD TP": od["TP"], "OD FP": od["FP"], "OD FN": od["FN"], "OD TN": od["TN"],
                
                "Infra Precision": infra["P"], "Infra Recall": infra["R"], "Infra F1": infra["F1"],
                "Infra TP": infra["TP"], "Infra FP": infra["FP"], "Infra FN": infra["FN"], "Infra TN": infra["TN"],
                
                "General Precision": gen["P"], "General Recall": gen["R"], "General F1": gen["F1"],
                "General TP": gen["TP"], "General FP": gen["FP"], "General FN": gen["FN"], "General TN": gen["TN"],
                
                "Det Precision": det["P"], "Det Recall": det["R"], "Det F1": det["F1"],
                "Det TP": det["TP"], "Det FP": det["FP"], "Det FN": det["FN"], "Det TN": det["TN"],
            }
            summary_rows.append(row)
        except Exception as e:
            print(f"[-] Warning: Failed to parse {file_path}: {e}")

    df = pd.DataFrame(summary_rows)
    df.sort_values(by="Overall F1", ascending=False, inplace=True)

    csv_out = os.path.join(aggregated_out_dir, "baseline_comparative_summary.csv")
    json_out = os.path.join(aggregated_out_dir, "baseline_comparative_summary.json")

    df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n[SUCCESS] Aggregated report generated in:\n    --> {aggregated_out_dir}\n")
    
    print(f"[*] Generating all requested visualization charts...")
    generate_all_visualizations(df, aggregated_out_dir)

    display_cols = [
        "Model Key", "Overall F1", "Overall TP", "Overall FP", 
        "Overall FN", "Overall TN", "Det F1", "OD F1", "Infra F1", "General F1"
    ]
    print("\n" + df[display_cols].to_string(index=False) + "\n")

    return aggregated_out_dir


if __name__ == "__main__":
    aggregate_all_benchmarks()