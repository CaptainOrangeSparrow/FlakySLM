# FlakySLM
This project aims to use small language models (SLMs) to detect python flaky test from source code.

# Dataset Creation and Description
This dataset is created to help SLMs learn how to spot flaky tests in Python code.

## Where does the data come from?
Tests come from the [International Dataset of Flaky Tests (IDoFT)](https://github.com/TestingResearchIllinois/idoft). Specifically, we look at the python related flaky tests, [py-data.csv](https://github.com/TestingResearchIllinois/idoft/blob/main/py-data.csv).
The team is also getting tests from the ["An Empirical Study of Flaky Tests in Python" dataset](https://zenodo.org/records/4450435).

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
