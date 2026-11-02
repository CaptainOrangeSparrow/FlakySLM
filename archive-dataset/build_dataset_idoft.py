import ast
import json
import os
import time
import pandas as pd
import requests

# configuration
IDOF_CSV_PATH = "py-data.csv"
OUTPUT_JSONL = "flaky_slm_dataset.jsonl"

# MAX_SAMPLES = 10
MAX_SAMPLES = None # runs through all py-data.csv tests

# git token, do not expose this so use ENV variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def get_headers():
  headers = {}
  if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"
  return headers


def load_existing_checkpoints(output_path: str) -> set:
  processed = set()
  if os.path.exists(output_path):
    print("found existing dataset checkpoint. Loading processed records...")
    with open(output_path, "r", encoding="utf-8") as f:
      for line in f:
        try:
          data = json.loads(line)
          if "test_identifier" in data:
            processed.add(data["test_identifier"])
        except json.JSONDecodeError:
          continue
    print(f"resumed: {len(processed)} records already completed.")
  return processed


def fetch_remote_file(project_url: str, sha: str, file_path: str) -> str:
  repo_path = project_url.replace("https://github.com/", "").rstrip("/")
  raw_url = f"https://raw.githubusercontent.com/{repo_path}/{sha}/{file_path}"

  response = requests.get(raw_url, headers=get_headers())

  if response.status_code == 200:
    return response.text
  else:
    print(
        f"Fetch failed (HTTP {response.status_code}) for URL: {raw_url}"
    )
    return None


def extract_functions_via_ast(source_code: str, target_func_name: str):
  try:
    tree = ast.parse(source_code)
  except SyntaxError as e:
    print(f"AST SyntaxError: {e}")
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
  if not os.path.exists(IDOF_CSV_PATH):
    print(f"Error: {IDOF_CSV_PATH} not found in working directory.")
    return

  processed_identifiers = load_existing_checkpoints(OUTPUT_JSONL)
  df = pd.read_csv(IDOF_CSV_PATH)

  print(
      f"Loaded CSV with {len(df)} rows. Target sample limit: {MAX_SAMPLES}"
  )

  processed_count = 0

  with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
    for idx, row in df.iterrows():
      if MAX_SAMPLES and processed_count >= MAX_SAMPLES:
        print(f"Reached target limit of {MAX_SAMPLES} successful samples.")
        break

      project_url = str(row.get("Project URL", "")).strip()
      sha = str(row.get("SHA Detected", "")).strip()

      # do the exact match for the CSV's long column name
      test_identifier = str(
          row.get(
              "Pytest Test Name (PathToFile::TestClass::TestMethod or"
              " PathToFile::TestMethod)",
              "",
          )
      ).strip()
      category = str(row.get("Category", "Unknown")).strip()

      if not project_url or not sha or not test_identifier:
        print(f"skipping row {idx}: Misssing metadata.")
        continue

      if test_identifier in processed_identifiers:
        continue

      # idoft format can be Path::Class::Method or Path::Method
      parts = test_identifier.split("::")
      file_path = parts[0]
      target_func = parts[-1]  # aactual test method name is always last

      # fetch source code
      source_code = fetch_remote_file(project_url, sha, file_path)
      if not source_code:
        continue

      # extractt via AST
      flaky_code, siblings = extract_functions_via_ast(source_code, target_func)
      if not flaky_code:
        continue

      # create flaky record
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
                      f" '{category}' based on historical IDoFT execution"
                      " data."
                  ),
              },
          ],
      }
      f_out.write(json.dumps(flaky_record) + "\n")

      # create non-flaky record
      if siblings:
        sib_name, sib_code = siblings[0]
        non_flaky_record = {
            "project_url": project_url,
            "test_identifier": f"{file_path}::{sib_name}",
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
          f"YO ({processed_count}) Successfully Processed: {target_func} +"
          " Sibling Control"
      )

  print(f"\nDone. Data pipelne complete. Check {OUTPUT_JSONL}")


if __name__ == "__main__":
  build_dataset()
