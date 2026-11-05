import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import config

# Import vLLM components
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("[-] Error: 'vllm' package not found. Please install via: pip install vllm")
    exit(1)


# =====================================================================
# 1. HIERARCHICAL FALLBACK PARSER
# =====================================================================
def parse_model_prediction(raw_response: str) -> Tuple[str, bool]:
    """
    Parses LLM output into 'flaky' or 'non_flaky'.
    Returns (prediction, parsing_error_flag).
    """
    if not raw_response or not raw_response.strip():
        return "non_flaky", True

    text = raw_response.lower()

    # Step 1: Check negative indicators FIRST (e.g., "non-flaky", "deterministic")
    for pattern in config.NEGATIVE_PATTERNS:
        if re.search(pattern, text):
            return "non_flaky", False

    # Step 2: Check positive indicators (e.g., "flaky")
    for pattern in config.POSITIVE_PATTERNS:
        if re.search(pattern, text):
            return "flaky", False

    # Step 3: Fallback if ambiguous output
    return "non_flaky", True


# =====================================================================
# 2. METRIC CALCULATION ENGINE
# =====================================================================
def compute_classification_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


# =====================================================================
# 3. MAIN EVALUATION PIPELINE
# =====================================================================
def run_evaluation(run_tag: str = "baseline"):
    print(f"=====================================================")
    print(f"   STARTING BASELINE EVALUATION: {config.ACTIVE_MODEL_KEY}")
    print(f"=====================================================")
    
    if not os.path.exists(config.TEST_DATASET_PATH):
        print(f"[-] Error: Test dataset file not found at '{config.TEST_DATASET_PATH}'")
        return

    run_output_dir = config.get_run_output_dir(run_tag)
    predictions_log_path = os.path.join(run_output_dir, "predictions_log.jsonl")
    scorecard_summary_path = os.path.join(run_output_dir, "scorecard_summary.json")

    # Load dataset_test.jsonl
    test_records = []
    with open(config.TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))
    print(f"[*] Loaded {len(test_records):,} test records from {config.TEST_DATASET_PATH}")

    # Prepare vLLM Prompts
    prompts = []
    for rec in test_records:
        # Construct standard conversation format
        messages = rec.get("messages", [])
        # Extract system and user messages if present
        prompt_text = ""
        for m in messages:
            if m["role"] == "system":
                prompt_text += f"System: {m['content']}\n\n"
            elif m["role"] == "user":
                prompt_text += f"User: {m['content']}\n\nAssistant:"
        if not prompt_text:
            prompt_text = f"Analyze this Python test function for flakiness:\n\n{rec.get('code', '')}\n\nAssistant:"
        prompts.append(prompt_text)

    # Initialize vLLM Model
    print(f"[*] Initializing vLLM engine for model: {config.MODEL_NAME}...")
    llm = LLM(
        model=config.MODEL_NAME,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_model_len=config.MAX_MODEL_LEN,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        max_tokens=config.MAX_TOKENS
    )

    # Execute Inference & Record Execution Latency
    print(f"[*] Executing batch zero-shot inference on {len(prompts)} samples...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_batch_latency = time.time() - start_time

    # Process & Score Outputs
    detailed_logs = []
    category_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    overall_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    parsing_error_count = 0

    for idx, (rec, output) in enumerate(zip(test_records, outputs)):
        generated_text = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        # Ground truth label
        ground_truth = rec.get("label", "non_flaky").lower()
        sub_category = rec.get("metadata", {}).get("sub_category", "General Flakiness")
        if ground_truth == "non_flaky":
            sub_category = "None (Deterministic)"

        # Parse LLM prediction
        pred_label, parsing_error = parse_model_prediction(generated_text)
        if parsing_error:
            parsing_error_count += 1

        # Binary confusion matrix assignment
        is_tp = (ground_truth == "flaky" and pred_label == "flaky")
        is_fp = (ground_truth == "non_flaky" and pred_label == "flaky")
        is_fn = (ground_truth == "flaky" and pred_label == "non_flaky")
        is_tn = (ground_truth == "non_flaky" and pred_label == "non_flaky")

        if is_tp:
            overall_counts["tp"] += 1
            category_counts[sub_category]["tp"] += 1
        elif is_fp:
            overall_counts["fp"] += 1
            category_counts[sub_category]["fp"] += 1
        elif is_fn:
            overall_counts["fn"] += 1
            category_counts[sub_category]["fn"] += 1
        elif is_tn:
            overall_counts["tn"] += 1
            category_counts[sub_category]["tn"] += 1

        log_item = {
            "sample_index": idx,
            "project_url": rec.get("project_url", ""),
            "test_identifier": rec.get("test_identifier", ""),
            "ground_truth": ground_truth,
            "predicted_label": pred_label,
            "sub_category": sub_category,
            "parsing_error": parsing_error,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "raw_output": generated_text
        }
        detailed_logs.append(log_item)

    # Save detailed prediction logs line-by-line
    with open(predictions_log_path, "w", encoding="utf-8") as f:
        for log in detailed_logs:
            f.write(json.dumps(log) + "\n")

    # Calculate final disaggregated metrics
    overall_metrics = compute_classification_metrics(
        overall_counts["tp"], overall_counts["fp"], overall_counts["fn"], overall_counts["tn"]
    )

    disaggregated_category_metrics = {}
    for cat_name, counts in category_counts.items():
        disaggregated_category_metrics[cat_name] = compute_classification_metrics(
            counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        )

    avg_latency_per_sample = total_batch_latency / len(test_records) if test_records else 0.0

    summary_scorecard = {
        "model_name": config.MODEL_NAME,
        "model_key": config.ACTIVE_MODEL_KEY,
        "run_tag": run_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(test_records),
        "parsing_errors": parsing_error_count,
        "operational_metrics": {
            "total_inference_time_sec": round(total_batch_latency, 2),
            "avg_latency_per_sample_sec": round(avg_latency_per_sample, 4),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens_consumed": total_prompt_tokens + total_completion_tokens,
            "avg_tokens_per_sample": round((total_prompt_tokens + total_completion_tokens) / len(test_records), 2)
        },
        "overall_classification_metrics": overall_metrics,
        "disaggregated_sub_category_metrics": disaggregated_category_metrics
    }

    # Save scorecard summary JSON
    with open(scorecard_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_scorecard, f, indent=2)

    # Display final summary on console
    print("\n====================================================")
    print("           BASELINE EVALUATION COMPLETE             ")
    print("====================================================")
    print(f" Model Key      : {config.ACTIVE_MODEL_KEY}")
    print(f" Saved Results  : {run_output_dir}")
    print(f" Overall F1     : {overall_metrics['f1_score']}")
    print(f" Precision      : {overall_metrics['precision']}")
    print(f" Recall         : {overall_metrics['recall']}")
    print(f" Accuracy       : {overall_metrics['accuracy']}")
    print(f" Total Time     : {round(total_batch_latency, 2)}s (Avg {round(avg_latency_per_sample, 3)}s/sample)")
    print(f" Parsing Errors : {parsing_error_count} / {len(test_records)}")
    print("====================================================\n")

if __name__ == "__main__":
    run_evaluation(run_tag="zero_shot_baseline")