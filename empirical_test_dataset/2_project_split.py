import json
import os
import random

INPUT_RAW_PAIRS = "extracted_raw_pairs.json"
TRAIN_OUT = "split_train_raw.json"
VAL_OUT = "split_val_raw.json"
TEST_OUT = "split_test_raw.json"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def main():
    print("--- SCRIPT 2: Performing Project-Disjoint Split ---")
    if not os.path.exists(INPUT_RAW_PAIRS):
        print(f"[-] Error: {INPUT_RAW_PAIRS} not found. Please run Script 1 first.")
        return

    with open(INPUT_RAW_PAIRS, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"[*] Loaded {len(pairs):,} raw test pairs.")

    # Group pairs by project URL
    project_map = {}
    for pair in pairs:
        p_url = pair["project_url"]
        if p_url not in project_map:
            project_map[p_url] = []
        project_map[p_url].append(pair)

    unique_projects = list(project_map.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(unique_projects)

    total_projects = len(unique_projects)
    train_end = int(total_projects * TRAIN_RATIO)
    val_end = train_end + int(total_projects * VAL_RATIO)

    train_projects = set(unique_projects[:train_end])
    val_projects = set(unique_projects[train_end:val_end])
    test_projects = set(unique_projects[val_end:])

    train_data, val_data, test_data = [], [], []

    for p_url, p_pairs in project_map.items():
        if p_url in train_projects:
            train_data.extend(p_pairs)
        elif p_url in val_projects:
            val_data.extend(p_pairs)
        elif p_url in test_projects:
            test_data.extend(p_pairs)

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(VAL_OUT, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"[*] Total Unique Repositories: {total_projects}")
    print(f"[+] Train Split: {len(train_projects)} repos, {len(train_data):,} records saved to {TRAIN_OUT}")
    print(f"[+] Validation Split: {len(val_projects)} repos, {len(val_data):,} records saved to {VAL_OUT}")
    print(f"[+] Test Split: {len(test_projects)} repos, {len(test_data):,} records saved to {TEST_OUT}")
    print("[SUCCESS] Script 2 complete.")

if __name__ == "__main__":
    main()