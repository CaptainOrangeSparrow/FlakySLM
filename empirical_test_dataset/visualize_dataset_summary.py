import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
SPLITS = {
    "Train (70%)": "dataset_train.jsonl",
    "Validation (15%)": "dataset_val.jsonl",
    "Test (15%)": "dataset_test.jsonl",
}

OUTPUT_DIR = "dataset_report_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_jsonl_records(filepath: str) -> list:
    records = []
    if not os.path.exists(filepath):
        print(f"[-] Warning: {filepath} not found.")
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze_and_visualize():
    print("=====================================================")
    print("   DATASET SPLIT & SUB-CATEGORY ANALYSIS REPORT")
    print("=====================================================")

    all_rows = []
    summary_table_rows = []

    for split_name, filepath in SPLITS.items():
        records = load_jsonl_records(filepath)
        if not records:
            continue

        unique_repos = set()
        flaky_count = 0
        non_flaky_count = 0
        subcat_counts = {
            "Order-Dependent (OD)": 0,
            "Infrastructure / Environment": 0,
            "General Flakiness": 0,
            "None (Deterministic)": 0,
        }

        for rec in records:
            p_url = rec.get("project_url", "")
            unique_repos.add(p_url)
            label = rec.get("label", "")
            meta = rec.get("metadata", {})
            raw_subcat = meta.get("sub_category", "None")

            if label == "flaky":
                flaky_count += 1
                subcat = raw_subcat if raw_subcat in subcat_counts else "General Flakiness"
            else:
                non_flaky_count += 1
                subcat = "None (Deterministic)"

            subcat_counts[subcat] += 1
            all_rows.append({
                "Split": split_name,
                "Label": "Flaky" if label == "flaky" else "Non-Flaky",
                "Sub-Category": subcat,
                "Project": p_url,
            })

        total_samples = len(records)
        summary_table_rows.append({
            "Dataset Split": split_name,
            "Unique Repositories": len(unique_repos),
            "Total Test Cases": total_samples,
            "Flaky Tests": flaky_count,
            "Non-Flaky Controls": non_flaky_count,
            "Order-Dependent (OD)": subcat_counts["Order-Dependent (OD)"],
            "Infrastructure / Env": subcat_counts["Infrastructure / Environment"],
            "General Flakiness": subcat_counts["General Flakiness"],
        })

    if not summary_table_rows:
        print("[-] No dataset files found to analyze. Ensure JSONL files exist in the working directory.")
        return

    # 1. Create & Save Comprehensive Summary Table
    summary_df = pd.DataFrame(summary_table_rows)

    # Add a Total row
    totals = {
        "Dataset Split": "Total (100%)",
        "Unique Repositories": summary_df["Unique Repositories"].sum(),
        "Total Test Cases": summary_df["Total Test Cases"].sum(),
        "Flaky Tests": summary_df["Flaky Tests"].sum(),
        "Non-Flaky Controls": summary_df["Non-Flaky Controls"].sum(),
        "Order-Dependent (OD)": summary_df["Order-Dependent (OD)"].sum(),
        "Infrastructure / Env": summary_df["Infrastructure / Env"].sum(),
        "General Flakiness": summary_df["General Flakiness"].sum(),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)

    csv_path = os.path.join(OUTPUT_DIR, "dataset_summary_table.csv")
    md_path = os.path.join(OUTPUT_DIR, "dataset_summary_table.md")
    summary_df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_df.to_markdown(index=False))

    print("\n--- DATASET SUMMARY TABLE ---")
    print(summary_df.to_markdown(index=False))
    print(f"\n[+] Saved CSV table to: {csv_path}")
    print(f"[+] Saved Markdown table to: {md_path}")

    # 2. Visualizations
    full_df = pd.DataFrame(all_rows)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Plot A: Split Distribution (Flaky vs Non-Flaky & Unique Repos)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    split_order = list(SPLITS.keys())
    sns.countplot(
        data=full_df,
        x="Split",
        hue="Label",
        order=split_order,
        palette=["#d92d02", "#1b829e"],
        ax=axes[0],
    )
    axes[0].set_title("Class Balance Across Project-Disjoint Splits", fontweight="bold")
    axes[0].set_xlabel("Dataset Split")
    axes[0].set_ylabel("Number of Test Cases")
    for container in axes[0].containers:
        axes[0].bar_label(container, padding=3, fontsize=10)

    # Plot B: Flaky Sub-Category Breakdown per Split
    flaky_only_df = full_df[full_df["Label"] == "Flaky"]
    subcat_order = ["Order-Dependent (OD)", "Infrastructure / Environment", "General Flakiness"]
    sns.countplot(
        data=flaky_only_df,
        x="Split",
        hue="Sub-Category",
        order=split_order,
        hue_order=subcat_order,
        palette="inferno",
        ax=axes[1],
    )
    axes[1].set_title("Flaky Test Sub-Category Distribution by Split", fontweight="bold")
    axes[1].set_xlabel("Dataset Split")
    axes[1].set_ylabel("Number of Flaky Test Cases")
    axes[1].legend(title="Flakiness Category", loc="upper right")
    for container in axes[1].containers:
        axes[1].bar_label(container, padding=3, fontsize=10)

    plt.tight_layout()
    combined_plot_path = os.path.join(OUTPUT_DIR, "dataset_splits_and_subcategories.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved publication chart to: {combined_plot_path}")


if __name__ == "__main__":
    analyze_and_visualize()