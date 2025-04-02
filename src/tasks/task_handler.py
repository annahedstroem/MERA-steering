from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datasets import load_dataset, load_from_disk
import numpy as np
import requests
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

hf_token = "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"


def generate_text_variants(word: str, remove_lower: bool = False) -> list:
    transformations = [word.lower(), word.upper(), word.capitalize()]
    if remove_lower:
        del transformations[0]
    spaces = [" ", ""]  # ' ', , '\n'
    return [
        f"{prefix}{variant}" for variant in transformations for prefix in spaces
    ]  # for suffix in spaces {suffix}


dataset_info = {
    "sentiment_analysis": {
        "CLASSES": ["positive", "neutral", "negative"],
        "CLASS_LABEL_TO_INDEX": {"positive": 0, "neutral": 1, "negative": 2},
        "CLASS_INDEX_TO_LABEL": {0: "positive", 1: "neutral", 2: "negative"},
        "CLASS_LABEL_SEMANTIC": {
            "positive": generate_text_variants("positive"),
            "neutral": generate_text_variants("neutral"),
            "negative": generate_text_variants("negative"),
        },
        "DATASET_NAME": "finance-instruct",
        "MAX_LENGTH": 150,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "NR_CALIBRATION_SAMPLES": 3000,
        "NR_REF_SAMPLES": 250,
        "NR_TEST_SAMPLES": 250,
    },
    "yes_no_question": {
        "CLASSES": ["Yes", "No"],
        "CLASS_LABEL_TO_INDEX": {"Yes": 0, "No": 1},
        "CLASS_INDEX_TO_LABEL": {0: "Yes", 1: "No"},
        "CLASS_LABEL_SEMANTIC": {
            "Yes": generate_text_variants("Yes"),
            "No": generate_text_variants("No"),
        },
        "DATASET_NAME": "finance-instruct",
        "MAX_LENGTH": 350,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "NR_CALIBRATION_SAMPLES": 3000,
        "NR_REF_SAMPLES": 250,
        "NR_TEST_SAMPLES": 250,
    },
    "sms_spam": {
        "CLASSES": ["ham", "spam"],
        "CLASS_LABEL_TO_INDEX": {"ham": 0, "spam": 1},
        "CLASS_INDEX_TO_LABEL": {0: "ham", 1: "spam"},
        "CLASS_LABEL_SEMANTIC": {
            "ham": generate_text_variants("ham"),
            "spam": generate_text_variants("spam"),
        },
        "DATASET_NAME": "sms_spam",
        "MAX_LENGTH": 350,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "label",
        "NR_CALIBRATION_SAMPLES": 3000,
        "NR_REF_SAMPLES": 250,
        "NR_TEST_SAMPLES": 250,
    },
    "imdb": {
        "CLASSES": ["negative", "positive"],
        "CLASS_LABEL_TO_INDEX": {"positive": 1, "negative": 0},
        "CLASS_INDEX_TO_LABEL": {1: "positive", 0: "negative"},
        "CLASS_LABEL_SEMANTIC": {
            "positive": generate_text_variants("positive"),
            "negative": generate_text_variants("negative"),
        },
        "DATASET_NAME": "imdb",
        "MAX_LENGTH": 350,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "label",
        "NR_CALIBRATION_SAMPLES": 3000,
        "NR_REF_SAMPLES": 250,
        "NR_TEST_SAMPLES": 250,
    },
    "mmlu_pro_natural_science": {
        "CLASSES": [chr(i) for i in range(65, 91)],
        "CLASS_LABEL_TO_INDEX": {chr(i): i - 65 for i in range(65, 91)},
        "CLASS_INDEX_TO_LABEL": {i - 65: chr(i) for i in range(65, 91)},
        "CLASS_LABEL_SEMANTIC": {
            chr(i): generate_text_variants(chr(i), remove_lower=True)
            for i in range(65, 91)
        },
        "SUB_TASKS": ["computer science", "physics", "math", "engineering"],
        "DATASET_NAME": "mmlu-pro",
        "MAX_LENGTH": 150,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "NR_CALIBRATION_SAMPLES": 3000,
        "NR_REF_SAMPLES": 250,
        "NR_TEST_SAMPLES": 250,
    },
    "mmlu_high_school": {
        "CLASSES": [chr(i) for i in range(65, 69)],
        "CLASS_LABEL_TO_INDEX": {chr(i): i - 69 for i in range(65, 69)},
        "CLASS_INDEX_TO_LABEL": {i - 65: chr(i) for i in range(65, 69)},
        "CLASS_LABEL_SEMANTIC": {
            # chr(i): [f"{chr(i)}", f"{chr(i)} ", f" {chr(i)}"] for i in range(65, 69)
            chr(i): generate_text_variants(chr(i), remove_lower=True)
            for i in range(65, 69)
        },
        "SUB_TASKS": [
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
        ],
        "DATASET_NAME": "mmlu",
        "MAX_LENGTH": 250,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "NR_CALIBRATION_SAMPLES": 3000,
        "NR_REF_SAMPLES": 210,
        "NR_TEST_SAMPLES": 210,
    },
    "mmlu_professional": {
        "CLASSES": [chr(i) for i in range(65, 69)],
        "CLASS_LABEL_TO_INDEX": {chr(i): i - 65 for i in range(65, 69)},
        "CLASS_INDEX_TO_LABEL": {i - 65: chr(i) for i in range(65, 69)},
        "CLASS_LABEL_SEMANTIC": {
            chr(i): generate_text_variants(chr(i), remove_lower=True)
            for i in range(65, 69)
        },
        "DATASET_NAME": "mmlu",
        "MAX_LENGTH": 250,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "SUB_TASKS": [],  # dynamically selected
        "NR_CALIBRATION_SAMPLES": 2602,
        "NR_REF_SAMPLES": 210,
        "NR_TEST_SAMPLES": 210,
    },
}


@dataclass
class TaskConfig:
    """Assumes downloaded in cache dir."""

    token: str
    cache_dir: str
    dataset_name: str
    device: str
    nr_devices: int
    model_name: str
    flexible_match: bool
    batch_size: int = 1
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    nr_samples: Optional[int] = None
    nr_test_samples: Optional[int] = None
    nr_ref_samples: Optional[int] = None

    def __post_init__(self):
        print(f"[INFO] Initalising {self.dataset_name}")
        self.dataset_info = dataset_info[self.dataset_name]
        if self.nr_samples is None:
            self.nr_samples = self.dataset_info["NR_CALIBRATION_SAMPLES"]
        if self.nr_test_samples is None:
            self.nr_test_samples = self.dataset_info["NR_REF_SAMPLES"]
        if self.nr_ref_samples is None:
            self.nr_ref_samples = self.dataset_info["NR_TEST_SAMPLES"]

        self.dataset_name = self.dataset_info["DATASET_NAME"]
        self.model_kwargs = self.model_kwargs or {
            "token": self.token,
            "cache_dir": self.cache_dir,
            "device_map": "auto",
            "offload_folder": f"{self.cache_dir}/offload/",
            # "pad_token_id": tokenizer.eos_token_id,
            # "torch_dtype": torch.float16,
            # "device": self.device,
            # "nr_devices": self.nr_devices,
        }
        self.tokenizer_kwargs = self.tokenizer_kwargs or {
            "padding": False,
            "truncation": False,
            "return_tensors": "pt",
        }


class ModelHandler:
    def __init__(self, config: TaskConfig):
        print(f"[INFO] Loading model and tokenizer...")
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.nr_layers = len(self.model.model.layers)
        self.tokenizer_kwargs = self.config.tokenizer_kwargs
        print(f"[INFO] \n... Done.")

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, **self.config.model_kwargs
        )
        return tokenizer

    def _load_model(self):
        if self.config.model_name in [
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
        ]:
            self.config.model_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **self.config.model_kwargs
        )
        model.eval()
        return model


class DatasetHandler:
    def __init__(self, config: TaskConfig, tokenizer):
        print(f"[INFO] Loading dataset...")
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataset_info = self.config.dataset_info
        self._update_dataset_info_with_token_ids()

        self.ds = self._load_dataset()
        self.ds_samples = self._get_samples(end_idx=self.config.nr_samples, start_idx=0)
        self.y_true_labels, self.y_true = self._get_y_true_labels(self.ds_samples)
        self.prompts = self._get_prompts(self.ds_samples)

        if self.config.nr_test_samples is not None:
            end_idx = self.config.nr_samples + self.config.nr_test_samples
            start_idx = self.config.nr_samples
            self.ds_samples_test = self._get_samples(
                end_idx=end_idx, start_idx=start_idx
            )
            self.y_true_labels_test, self.y_true_test = self._get_y_true_labels(
                self.ds_samples_test
            )
            self.prompts_test = self._get_prompts(self.ds_samples_test)
        if self.config.nr_ref_samples is not None:
            end_idx = (
                self.config.nr_samples
                + self.config.nr_test_samples
                + self.config.nr_ref_samples
            )
            start_idx = self.config.nr_samples + self.config.nr_test_samples
            self.ds_samples_ref = self._get_samples(
                end_idx=end_idx, start_idx=start_idx
            )
            self.y_true_labels_ref, self.y_true_ref = self._get_y_true_labels(
                self.ds_samples_ref
            )
            self.prompts_ref = self._get_prompts(self.ds_samples_ref)
        print(f"[INFO] ... Done.")

    def _load_dataset(self):
        ds = load_from_disk(f"{self.config.cache_dir}{self.config.dataset_name}.hf")
        if self.config.dataset_name == "finance-instruct":
            ds = ds.filter(lambda x: x["task_type"] == self.config.dataset_name)
        elif self.config.dataset_name == "mmlu_pro":
            ds = ds.filter(lambda x: x["category"] in self.dataset_info["SUB_TASKS"])
        elif self.config.dataset_name == "mmlu":
            if self.config.dataset_name == "mmlu_professional":
                ds = ds.filter(lambda x: x["subject"].startswith("professional_"))
            else:
                ds = ds.filter(lambda x: x["subject"] in self.dataset_info["SUB_TASKS"])

        return ds

    def _get_samples(self, end_idx: int, start_idx: int = 0):
        if self.config.dataset_name in ["finance-instruct", "sms_spam"]:
            return self.ds["train"].select(range(start_idx, end_idx))
        elif self.config.dataset_name in ["mmlu_pro", "imdb"]:  # == "mmlu_pro":
            return self.ds.select(range(start_idx, end_idx))
        elif self.config.dataset_name in ["mmlu"]:  # , "imdb"]:
            return self.ds["test"].select(range(start_idx, end_idx))

    def _get_y_true_labels(self, samples):
        if self.config.dataset_name in ["mmlu", "sms_spam", "imdb"]:
            return [
                self.dataset_info["CLASS_INDEX_TO_LABEL"][
                    s[self.dataset_info["LABEL_NAME"]]
                ]
                for s in samples
            ], [s[self.dataset_info["LABEL_NAME"]] for s in samples]
        else:
            return [s[self.dataset_info["LABEL_NAME"]] for s in samples], [
                self.dataset_info["CLASS_LABEL_TO_INDEX"][
                    s[self.dataset_info["LABEL_NAME"]]
                ]
                for s in samples
            ]

    def _update_dataset_info_with_token_ids(self):
        """Updates `VALID_GROUND_TRUTH_TOKEN_IDS` with the last token ID for each semantic label variant."""
        semantic_variants = self.dataset_info.get("CLASS_LABEL_SEMANTIC", {})
        self.dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"] = {}

        for label, synonyms in semantic_variants.items():
            self.dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"][label] = {}
            for word in [label] + synonyms:
                tokenized_word = self.tokenizer([word], return_tensors="pt").input_ids[
                    0
                ]
                token_id = tokenized_word[-1].item()
                self.dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"][label][
                    word
                ] = token_id
                # print(f"{word}: {token_id}")

        print(
            f"[INFO] VALID_GROUND_TRUTH_TOKEN_IDS updated: {self.dataset_info['VALID_GROUND_TRUTH_TOKEN_IDS']}"
        )

    def _get_prompts(self, samples):
        if self.config.dataset_name == "finance-instruct":
            return [s["inputs"] + " " for s in samples]
        prompts = []

        if "mmlu" in self.config.dataset_name:
            if "pro" in self.config.dataset_name:
                options = samples["options"]
            else:
                options = samples["choices"]

            for i, option in enumerate(options):
                formatted_options = "\n".join(
                    [
                        f"{self.dataset_info['CLASS_INDEX_TO_LABEL'][i]}. {choice}"
                        for i, choice in enumerate(option)
                    ]
                )
                answer_options = ", ".join(
                    [
                        f"{self.dataset_info['CLASS_INDEX_TO_LABEL'][i]}"
                        for i in range(len(option))
                    ]
                )
                prompt = f"""Question: {samples['question'][i]}
                \nOptions:\n{formatted_options}
                \nPlease select the correct answer. 
                Only return one letter {answer_options}. 
                Answer:\n """
                prompts.append(prompt)

        elif self.config.dataset_name == "sms_spam":
            sms_inputs = [s["sms"] + " " for s in samples]
            for sms in sms_inputs:
                prompt = f"""
                This SMS (text message): "{sms.strip()}" is classified as either spam or ham.
                \nPlease evaluate the content of the SMS and select the correct classification.
                \nOnly return one word: "ham" or "spam".
                Answer:\n """
                prompts.append(prompt)

        elif self.config.dataset_name == "imdb":
            texts = [s["text"] + " " for s in samples]
            for text in texts:
                prompt = f"""
                Review: "{text.strip()}"
                \nPlease classify this review as either "positive or "negative" sentiment.
                \nOnly return one word: "positive" for a positive review or "negative" for a negative review.
                Answer:\n """
                prompts.append(prompt)

        return prompts

    def __len__(self):
        return len(self.ds_samples)
