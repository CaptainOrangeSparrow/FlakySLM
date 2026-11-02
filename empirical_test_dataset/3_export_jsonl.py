import json
import os

def convert_split_to_jsonl(input_filename, output_filename):
    if not os.path.exists(input_filename):
        print(f"[-] Warning: {input_filename} not found, skipping.")
        return

    with open(input_filename, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    jsonl_records = []
    for pair in pairs:
        # 1. Flaky Record
        flaky_item = pair["flaky"]
        sub_cat = flaky_item["metadata"]["sub_category"]
        flaky_record = {
            "project_url": pair["project_url"],
            "test_identifier": flaky_item["identifier"],
            "label": "flaky",
            "metadata": flaky_item["metadata"],
            "messages": [
                {"role": "system", "content": "You are an expert software engineer specializing in Python test quality and flakiness detection."},
                {"role": "user", "content": f"Analyze this Python pytest function for flakiness:\n\n```python\n{flaky_item['code']}\n```"},
                {"role": "assistant", "content": f"Classification: Flaky\nRoot Cause Category: {sub_cat}\nReasoning: This test exhibits non-deterministic behavior categorized under '{sub_cat}' based on historical empirical study execution data."}
            ]
        }
        jsonl_records.append(flaky_record)

        # 2. Non-Flaky Record
        nf_item = pair["non_flaky"]
        nf_record = {
            "project_url": pair["project_url"],
            "test_identifier": nf_item["identifier"],
            "label": "non_flaky",
            "metadata": nf_item["metadata"],
            "messages": [
                {"role": "system", "content": "You are an expert software engineer specializing in Python test quality and flakiness detection."},
                {"role": "user", "content": f"Analyze this Python pytest function for flakiness:\n\n```python\n{nf_item['code']}\n```"},
                {"role": "assistant", "content": "Classification: Non-Flaky\nRoot Cause Category: None\nReasoning: This test is fully deterministic with local assertions and no external state leakage, concurrency hooks, or timing dependencies."}
            ]
        }
        jsonl_records.append(nf_record)

    with open(output_filename, "w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(record) + "\n")

    print(f"[+] Converted {len(pairs):,} pairs into {len(jsonl_records):,} records saved to {output_filename}")

def main():
    print("--- SCRIPT 3: Exporting to Unsloth/LlamaFactory JSONL ---")
    convert_split_to_jsonl("split_train_raw.json", "dataset_train.jsonl")
    convert_split_to_jsonl("split_val_raw.json", "dataset_val.jsonl")
    convert_split_to_jsonl("split_test_raw.json", "dataset_test.jsonl")
    print("[SUCCESS] Script 3 complete. All JSONL files ready for fine-tuning, sir!")

if __name__ == "__main__":
    main()