import ast
import json
import os
import pandas as pd
import requests

# ==========================================
# CONFIGURATION
# ==========================================
EMPIRICAL_CSV_PATH = "empirical-testsoverview.csv"
OUTPUT_RAW_PAIRS = "extracted_raw_pairs.json"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Set to an integer (e.g., 50) for testing, or None for the full dataset run
MAX_PAIRS = None 
# MAX_PAIRS = 3

def get_headers():
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def fetch_remote_file(project_url: str, sha: str, file_path: str) -> str:
    """Fetches raw python file content from GitHub using commit SHA."""
    repo_path = project_url.replace("https://github.com/", "").rstrip("/")
    raw_url = f"https://raw.githubusercontent.com/{repo_path}/{sha}/{file_path}"
    try:
        response = requests.get(raw_url, headers=get_headers(), timeout=15)
        if response.status_code == 200:
            return response.text
    except Exception:
        pass
    return None

def extract_function_via_ast(source_code: str, target_func_name: str):
    """Parses source code via AST and returns the exact function source snippet."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None
    
    lines = source_code.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == target_func_name:
                if node.lineno <= len(lines) and node.end_lineno <= len(lines):
                    return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None

def main():
    print("--- SCRIPT 1: Extracting Flaky & Project-Matched Non-Flaky Pairs ---")
    if not os.path.exists(EMPIRICAL_CSV_PATH):
        print(f"[-] Error: {EMPIRICAL_CSV_PATH} not found in working directory.")
        return

    df = pd.read_csv(EMPIRICAL_CSV_PATH)
    print(f"[*] Loaded master CSV with {len(df):,} total rows.")

    flaky_cols = [
        'Flaky_sameOrder_withinIteration',
        'Flaky_randomOrder_withinIteration',
        'Order-dependent',
        'Flaky_Infrastructure'
    ]
    for col in flaky_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower() == 'true'

    is_flaky = df[flaky_cols].any(axis=1)
    flaky_df = df[is_flaky]
    non_flaky_df = df[~is_flaky]

    print(f"[*] Indexing non-flaky candidates by project URL...")
    non_flaky_by_project = non_flaky_df.groupby('Project_URL')

    collected_pairs = []
    processed = 0

    for idx, flaky_row in flaky_df.iterrows():
        if MAX_PAIRS and processed >= MAX_PAIRS:
            print(f"[*] Reached target sample limit of {MAX_PAIRS}.")
            break

        project_url = str(flaky_row.get("Project_URL", "")).strip()
        sha = str(flaky_row.get("Project_Hash", "")).strip()
        file_path = str(flaky_row.get("Test_filename", "")).strip()
        class_name = str(flaky_row.get("Test_classname", "")).strip()
        target_func = str(flaky_row.get("Test_funcname", "")).strip()
        project_name = str(flaky_row.get("Project_Name", "")).strip()

        if not project_url or not sha or not file_path or not target_func:
            continue

        # Determine sub-category for metadata tracking
        if flaky_row.get("Order-dependent", False):
            sub_cat = "Order-Dependent (OD)"
        elif flaky_row.get("Flaky_Infrastructure", False):
            sub_cat = "Infrastructure / Environment"
        else:
            sub_cat = "General Flakiness"

        # 1. Fetch Flaky Source Code
        source_code = fetch_remote_file(project_url, sha, file_path)
        if not source_code:
            continue
        flaky_code = extract_function_via_ast(source_code, target_func)
        if not flaky_code:
            continue

        # 2. Find a Project-Matched Non-Flaky Test from the Same Repo
        if project_url not in non_flaky_by_project.groups:
            continue
        
        nf_candidates = non_flaky_by_project.get_group(project_url)
        nf_code = None
        nf_metadata = None

        for _, nf_row in nf_candidates.iterrows():
            nf_file = str(nf_row.get("Test_filename", "")).strip()
            nf_func = str(nf_row.get("Test_funcname", "")).strip()
            if not nf_file or not nf_func:
                continue
            
            nf_source = fetch_remote_file(project_url, sha, nf_file)
            if not nf_source:
                continue
            extracted_nf = extract_function_via_ast(nf_source, nf_func)
            if extracted_nf:
                nf_code = extracted_nf
                nf_class = str(nf_row.get("Test_classname", "")).strip()
                nf_metadata = {
                    "project_name": project_name,
                    "project_url": project_url,
                    "project_hash": sha,
                    "test_filename": nf_file,
                    "test_classname": nf_class if nf_class and nf_class.lower() not in ["nan", "none", ""] else "",
                    "test_funcname": nf_func,
                    "sub_category": "None"
                }
                break

        if not nf_code:
            continue

        flaky_identifier = f"{file_path}::{class_name}::{target_func}" if class_name and class_name.lower() not in ["nan", "none", ""] else f"{file_path}::{target_func}"
        nf_identifier = f"{nf_metadata['test_filename']}::{nf_metadata['test_classname']}::{nf_metadata['test_funcname']}" if nf_metadata['test_classname'] else f"{nf_metadata['test_filename']}::{nf_metadata['test_funcname']}"

        flaky_meta = {
            "project_name": project_name,
            "project_url": project_url,
            "project_hash": sha,
            "test_filename": file_path,
            "test_classname": class_name if class_name and class_name.lower() not in ["nan", "none", ""] else "",
            "test_funcname": target_func,
            "sub_category": sub_cat
        }

        pair = {
            "project_url": project_url,
            "flaky": {
                "identifier": flaky_identifier,
                "code": flaky_code,
                "metadata": flaky_meta
            },
            "non_flaky": {
                "identifier": nf_identifier,
                "code": nf_code,
                "metadata": nf_metadata
            }
        }

        collected_pairs.append(pair)
        processed += 1
        print(f"[+] Paired #{processed}: {target_func} (Repo: {project_name})")

    with open(OUTPUT_RAW_PAIRS, "w", encoding="utf-8") as f:
        json.dump(collected_pairs, f, indent=2)
    print(f"\n[SUCCESS] Script 1 complete. Saved {len(collected_pairs):,} pairs to {OUTPUT_RAW_PAIRS}")

if __name__ == "__main__":
    main()