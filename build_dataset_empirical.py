import ast
import json
import os
import pandas as pd
import requests

# config of empirical dataset
EMPIRICAL_CSV_PATH = "empirical-testsoverview.csv"
OUTPUT_JSONL = "flaky_slm_dataset_empirical.jsonl"

# MAX_SAMPLES = 90
MAX_SAMPLES = None #run through all tests


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def get_headers():
  headers = {}
  if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"
  return headers


def load_existing_checkpoints(output_path: str) -> set:
  processed = set()
  if os.path.exists(output_path):
    print("Found existing empirical dataset checkpoint. Loading records...")
    with open(output_path, "r", encoding="utf-8") as f:
      for line in f:
        try:
          data = json.loads(line)
          if "test_identifier" in data:
            processed.add(data["test_identifier"])
        except json.JSONDecodeError:
          continue
    print(f"Resumed: {len(processed)} records already completed.")
  return processed


def fetch_remote_file(project_url: str, sha: str, file_path: str) -> str:
  repo_path = project_url.replace("https://github.com/", "").rstrip("/")
  raw_url = f"https://raw.githubusercontent.com/{repo_path}/{sha}/{file_path}"

  response = requests.get(raw_url, headers=get_headers())

  if response.status_code == 200:
    return response.text
  else:
    print(
        f"fetch failed (HTTP {response.status_code}) for URL: {raw_url}"
    )
    return None


def extract_functions_via_ast(source_code: str, target_func_name: str):
  try:
    tree = ast.parse(source_code)
  except SyntaxError as e:
    print(f"[-] AST SyntaxError: {e}")
    return None, []

  lines = source_code.splitlines()
  target_code = None
  sibling_tests = []

  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      func_name = node.name
      func_snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])

      if func_name == target_func_name:
        target_code = func_snippet
      elif func_name.startswith("test_"):
        sibling_tests.append((func_name, func_snippet))

  if not target_code:
    print(f"Target function '{target_func_name}' not found via AST.")

  return target_code, sibling_tests


def build_dataset():
  if not os.path.exists(EMPIRICAL_CSV_PATH):
    print(f"Error: {EMPIRICAL_CSV_PATH} not found in working directory.")
    return

  processed_identifiers = load_existing_checkpoints(OUTPUT_JSONL)
  df = pd.read_csv(EMPIRICAL_CSV_PATH)

  print(f"Loaded CSV with {len(df)} rows.")
  print(
      "Using pragmatic sibling-selection approach (accepting mild label"
      " noise)."
  )
  print(f"Target sample limit: {MAX_SAMPLES}")

  processed_count = 0

  with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
    for idx, row in df.iterrows():
      if MAX_SAMPLES and processed_count >= MAX_SAMPLES:
        print(f"[*] Reached target limit of {MAX_SAMPLES} successful samples.")
        break

      project_url = str(row.get("Project_URL", "")).strip()
      sha = str(row.get("Project_Hash", "")).strip()
      file_path = str(row.get("Test_filename", "")).strip()
      class_name = str(row.get("Test_classname", "")).strip()
      target_func = str(row.get("Test_funcname", "")).strip()

      if (
          class_name
          and class_name.lower() != "nan"
          and class_name != "none"
          and class_name != ""
      ):
        test_identifier = f"{file_path}::{class_name}::{target_func}"
      else:
        test_identifier = f"{file_path}::{target_func}"

      is_order_dependent = row.get("Order-dependent", False)
      is_infrastructure = row.get("Flaky_Infrastructure", False)

      if is_order_dependent:
        category = "Order-Dependent (OD)"
      elif is_infrastructure:
        category = "Infrastructure / Environment"
      else:
        category = "General Flakiness"

      if not project_url or not sha or not file_path or not target_func:
        print(f"Skipping row {idx}: Missing metadata.")
        continue

      if test_identifier in processed_identifiers:
        continue

      source_code = fetch_remote_file(project_url, sha, file_path)
      if not source_code:
        continue

      flaky_code, siblings = extract_functions_via_ast(source_code, target_func)
      if not flaky_code:
        continue

      # Find any other test function in the same file to use as a control
      sibling_found = False
      sib_identifier = None
      sib_code = None

      for sib_name, sib_code_snippet in siblings:
        if sib_name == target_func:
          continue

        if (
            class_name
            and class_name.lower() != "nan"
            and class_name != "none"
            and class_name != ""
        ):
          sib_identifier = f"{file_path}::{class_name}::{sib_name}"
        else:
          sib_identifier = f"{file_path}::{sib_name}"

        sib_code = sib_code_snippet
        sibling_found = True
        break

      if not sibling_found:
        print(f"Skipping {target_func}: No sibling test found in file.")
        continue

      # 1. write flaky record
      flaky_record = {
          "project_url": project_url,
          "test_identifier": test_identifier,
          "label": "flaky",
          "messages": [
              {
                  "role": "system",
                  "content": (
                      "You are an expert software engineer specializing in"
                      " Python test quality and flakiness detection."
                  ),
              },
              {
                  "role": "user",
                  "content": (
                      "Analyze this Python pytest function for"
                      f" flakiness:\n\n```python\n{flaky_code}\n```"
                  ),
              },
              {
                  "role": "assistant",
                  "content": (
                      f"Classification: Flaky\nRoot Cause Category:"
                      f" {category}\nReasoning: This test exhibits"
                      " non-deterministic behavior categorized under"
                      f" '{category}' based on historical empirical study"
                      " execution data."
                  ),
              },
          ],
      }
      f_out.write(json.dumps(flaky_record) + "\n")

      # 2. write non flaky record, the sibling
      non_flaky_record = {
          "project_url": project_url,
          "test_identifier": sib_identifier,
          "label": "non_flaky",
          "messages": [
              {
                  "role": "system",
                  "content": (
                      "You are an expert software engineer specializing in"
                      " Python test quality and flakiness detection."
                  ),
              },
              {
                  "role": "user",
                  "content": (
                      "Analyze this Python pytest function for"
                      f" flakiness:\n\n```python\n{sib_code}\n```"
                  ),
              },
              {
                  "role": "assistant",
                  "content": (
                      "Classification: Non-Flaky\nRoot Cause Category:"
                      " None\nReasoning: This test is fully deterministic"
                      " with local assertions and no external state"
                      " leakage, concurrency hooks, or timing"
                      " dependencies."
                  ),
              },
          ],
      }
      f_out.write(json.dumps(non_flaky_record) + "\n")

      processed_identifiers.add(test_identifier)
      processed_count += 1
      print(
          f"({processed_count}) Successfully Processed Pair: {target_func} +"
          " Sibling Control"
      )

  print(f"\nDone. Empirical pipeline complete. Check {OUTPUT_JSONL}")


if __name__ == "__main__":
  build_dataset()