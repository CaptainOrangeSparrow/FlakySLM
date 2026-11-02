# FlakySLM
This project aims to use small language models (SLMs) to detect python flaky test from source code.

# Dataset Creation and Description
This dataset is created to help SLMs learn how to spot flaky tests in Python code.

## Where does the data come from?
Tests come from the the ["An Empirical Study of Flaky Tests in Python" dataset](https://zenodo.org/records/4450435). They are python related flaky tests.

_(Future Work)_
The team may incorporate tests from the [International Dataset of Flaky Tests (IDoFT)](https://github.com/TestingResearchIllinois/idoft) for additional training or evaluation if time permits. Specifically, we look at the python related flaky tests, [py-data.csv](https://github.com/TestingResearchIllinois/idoft/blob/main/py-data.csv).

[START moby to update this section START]

To make sure our model does not experience total class collapse during fine-tuning (ending up predicting flaky test for every input), we collect flaky and non-flaky tests from the same project code files. Our current `build_dataset_idoft.py` and `build_dataset_empirical.py` files uses Python's native `ast` parser to find boundaries of each target function along with any associated decorators and inline assertions. We currently avoid reading whole files or entire code structure.

## Dataset Formatting
The dataset is serialized in JSON Lines (`.jsonl`) format which is compatabile with the [ChatML](https://huggingface.co/learn/llm-course/en/chapter11/2) instruction/conversation schema required by the fine-tuning libraries (e.g. Hugging Face SFTTrainer, Unsloth, Llama Factory).

Each line represents an independent JSON object structured as follows:
```json
{
  "project_url": "https://github.com/abeelen/nikamap", 
  "test_identifier": "nikamap/tests/test_nikamap.py::test_nikamap_write", 
  "label": "flaky", 
  "messages": [
    {
      "role": "system", 
      "content": "You are an expert software engineer specializing in Python test quality and flakiness detection."
    }, 
    {
    "role": "user", 
    "content": "Analyze this Python pytest function for flakiness:\n\n```python\ndef test_nikamap_write(generate_fits):\n    filename = generate_fits\n\n    data = NikaMap.read(filename)\n    data_2mm = NikaMap.read(filename, band=\"2mm\")\n    data_1mm = NikaMap.read(filename, band=\"1mm\")\n\n    outfilename = filename.replace(\"map.fits\", \"map2.fits\")\n    data.write(outfilename)\n    data_1mm.write(outfilename, overwrite=True)\n    data_2mm.write(outfilename, append=True)\n```"
      }, 
      {
      "role": "assistant", 
      "content": "Classification: Flaky\nRoot Cause Category: NIO\nReasoning: This test exhibits non-deterministic behavior categorized under 'NIO' based on historical IDoFT execution data."
    }
  ]
}
```

[END moby to update this section END]


## /empirical_test_dataset
This directory contains the updated dataset for FlakySLM. It is composed of the Python flaky tests from "An Empirical Study of Flaky Tests in Python".
It has project-disjoint split into a train (70%), validation (15%), and test split (15%). So 70% of the projects are used as training data for the SLMs and the other 15%s are projects used for validation and testing. Keep in mind the actual number of tests in each train/val/test split varies as a project-disjoint split was performed.

(`dataset_train.jsonl`) used to train the SLM.

(`dataset_val.jsonl`) used to verify and adjust SLM parameters and training .

(`dataset_test.jsonl`) used to benchmark SLM performance.

#test