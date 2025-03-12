## How to extend datasets

To add a new dataset, follow these steps:

### Update `dataset_info`
Modify the `dataset_info` dictionary to include an entry for the new dataset, please define
- `CLASSES`: list of possible labels
- `CLASS_LABEL_TO_INDEX`: mapping from class label to index
- `CLASS_INDEX_TO_LABEL`: mapping from index to class label
- `CLASS_LABEL_SEMANTIC`: synonyms for each label
- `DATASET_NAME`: cached (saved) dataset name (as in `../hf-cache/` folder)
- `MAX_LENGTH`: maximum token length for full length (prompt+completion)
- `MAX_NEW_TOKENS`: maximum number of new generated tokens
- `LABEL_NAME`: Field in the dataset corresponding to labels
- `NR_CALIBRATION_SAMPLES`: number of calibration samples
- `NR_REF_SAMPLES`: number of reference samples
- `NR_TEST_SAMPLES`: number of test samples

### Update `DatasetHandler`

Ensure the dataset is correctly processed by updating the following methods

- `_load_dataset`: add dataset-specific filtering logic if needed
- `_get_samples`: define how samples are selected
- `_get_y_true_labels`: ensure the label extraction logic aligns with the dataset format
- `_update_dataset_info_with_token_ids`: generate token IDs for class labels (valid ground truth labels)
- `_get_prompts`: define the prompt format for the dataset

### Example dataset entry

```python
"new_dataset": {
    "CLASSES": ["class1", "class2"],
    "CLASS_LABEL_TO_INDEX": {"class1": 0, "class2": 1},
    "CLASS_INDEX_TO_LABEL": {0: "class1", 1: "class2"},
    "CLASS_LABEL_SEMANTIC": {
        "class1": generate_text_variants("class1"),
        "class2": generate_text_variants("class2"),
    },
    "DATASET_NAME": "new_dataset",
    "MAX_LENGTH": 250,
    "MAX_NEW_TOKENS": 100,
    "LABEL_NAME": "label",
    "NR_CALIBRATION_SAMPLES": 3000,
    "NR_REF_SAMPLES": 250,
    "NR_TEST_SAMPLES": 250,
}
```

### Final steps

To ensure your dataset is fully integrated into the pipeline, after updating the dataset configuration and handler

1. Save the dataset in `hf-cache` using `save_to_disk()` function
2. Run the modified script to ensure the dataset loads and processes correctly