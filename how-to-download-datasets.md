## How to download datasets

This repository relies on specific datasets that must be downloaded and stored in a designated cache folder (`hf-cache`) outside your project directory.

### Folder structure

```
root/
  MERA-steering/
  hf-cache/
```

### Installation

Ensure you have the `datasets` library installed:
```bash
pip install datasets
```

### Download script

Run the following script to download and cache the datasets

```python
from datasets import load_dataset

cache_directory = "../../hf-cache/"
datasets = {
    "sujet-ai/Sujet-Finance-Instruct-177k": "finance-instrunct.hf",
    "ucirvine/sms_spam": "sms_spam",
    "cais/mmlu": "mmlu"
}

for ds_source, ds_name in datasets.items():
    ds = load_dataset(ds_source)
    ds.save_to_disk(cache_directory + ds_name)
```

The `tasks/task_handler` will then take care of the rest of the processing.