# FlakySLM
This project aims to use small language models (SLMs) to detect python flaky test from source code.

# Dataset Creation and Description
This dataset is created to help SLMs learn how to spot flaky tests in Python code. We have used a repository-matched control sampling strategy such that for every flaky test function identified within a given repository, we query the empirical dataset for a confirmed non-flaky test case from the exact same project commit. This prevents the models from relying on project-level code style or repo-specific vocabulary/syntax to actually learn structural semantics of Python flaky tests. During the train-val-test dataset split, we also enforce a project-disjoint split such that tests found in a training data will not appear in the validation or test data to prevent data leakage.

## Where does the data come from?
Tests come from the the ["An Empirical Study of Flaky Tests in Python" dataset](https://zenodo.org/records/4450435). They are python related flaky tests.

_(Future Work)_
The team may incorporate tests from the [International Dataset of Flaky Tests (IDoFT)](https://github.com/TestingResearchIllinois/idoft) for additional training or evaluation if time permits. Specifically, we look at the python related flaky tests, [py-data.csv](https://github.com/TestingResearchIllinois/idoft/blob/main/py-data.csv).

To make sure our model does not experience total class collapse during fine-tuning (ending up predicting flaky test for every input), we collect flaky and non-flaky tests from the same project code files. Our current `empirical_test_dataset/1_extract_pairs.py` uses Python's native `ast` parser to find boundaries of each target function along with any associated decorators and inline assertions. We currently avoid reading whole files or entire code structure.

# Dataset Makeup
A test pair is defined as a flaky and non-flaky python test. Saved 7,027 pairs to extracted_raw_pairs.json

Total Unique Repositories: 904

[+] Train Split: 632 repos, 5,061 records saved to split_train_raw.json

[+] Validation Split: 135 repos, 814 records saved to split_val_raw.json

[+] Test Split: 137 repos, 1,152 records saved to split_test_raw.json

\

[+] Converted 5,061 pairs into 10,122 records saved to dataset_train.jsonl

[+] Converted 814 pairs into 1,628 records saved to dataset_val.jsonl

[+] Converted 1,152 pairs into 2,304 records saved to dataset_test.jsonl

# Dataset Formatting
The dataset is serialized in JSON Lines (`.jsonl`) format which is compatabile with the [ChatML](https://huggingface.co/learn/llm-course/en/chapter11/2) instruction/conversation schema required by the fine-tuning libraries (e.g. Hugging Face SFTTrainer, Unsloth, Llama Factory).

Each line represents an independent JSON object structured as follows:
```json
{
  "project_url": "https://github.com/example/repo",
  "test_identifier": "tests/test_core.py::TestClass::test_feature",
  "label": "flaky",
  "metadata": {
    "project_name": "example-repo",
    "project_url": "https://github.com/example/repo",
    "project_hash": "a1b2c3d4e5f6g7h8i9",
    "test_filename": "tests/test_core.py",
    "test_classname": "TestClass",
    "sub_category": "Order-Dependent (OD)"
  },
  "messages": [
    {
      "role": "system",
      "content": "You are an expert software engineer specializing in Python test quality and flakiness detection."
    },
    {
      "role": "user",
      "content": "Analyze this Python pytest function for flakiness:\n\n```python\ndef test_feature():\n    ...\n```"
    },
    {
      "role": "assistant",
      "content": "Classification: Flaky\nRoot Cause Category: Order-Dependent (OD)\nReasoning: This test exhibits non-deterministic behavior categorized under 'Order-Dependent (OD)' based on historical empirical study execution data."
    }
  ]
}
```


## /empirical_test_dataset
This directory contains the updated dataset for FlakySLM. It is composed of the Python flaky tests from "An Empirical Study of Flaky Tests in Python".
It has project-disjoint split into a train (70%), validation (15%), and test split (15%). So 70% of the projects are used as training data for the SLMs and the other 15%s are projects used for validation and testing. Keep in mind the actual number of tests in each train/val/test split varies as a project-disjoint split was performed.

(`dataset_train.jsonl`) used to train the SLM.

(`dataset_val.jsonl`) used to verify and adjust SLM parameters and training .

(`dataset_test.jsonl`) used to benchmark SLM performance.

#test
