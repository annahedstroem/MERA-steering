import json
import pickle
from typing import List, Optional, Callable, Tuple, Dict, Any, Type
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

from tasks.task_handler import *
from cache.cache_utils import *
from .base import *

DELTA_COLS = [
    "inner_evaluation/Delta Accuracy",
    "inner_evaluation/Delta Accuracy Exact",
    "inner_evaluation/Delta F1 Score",
    "inner_evaluation/Delta F1 Score Exact",
    "inner_evaluation/Delta Recall",
    "inner_evaluation/Delta Recall Exact",
    "inner_evaluation/Delta Precision",
    "inner_evaluation/Delta Precision Exact",
    "inner_evaluation/Delta Error",
    "inner_evaluation/Delta Error Exact",
    "inner_evaluation/Corrections Total",
    "inner_evaluation/Corrections Percentage",
    "inner_evaluation/Corrections Total Exact",
    "inner_evaluation/Corrections Percentage Exact",
]


def compute_error_metrics(targets, prefix="inner_evaluation/"):
    """Compute error-related metrics."""
    metrics = {}
    for suffix, suffix_load in {"": "", " Exact", : "_exact"}.items():
        error_values = 1 - np.array(targets[f"y_softmax{suffix_load}"])
        metrics.update({
            f"{prefix}Error{suffix}": np.mean(error_values),
            f"{prefix}Accuracy{suffix}": np.mean(targets[f"y_correct{suffix_load}"]),
            f"{prefix}Error{suffix} Std": np.std(error_values),
            f"{prefix}Accuracy{suffix} Std": np.std(targets[f"y_correct{suffix_load}"]),
            f"{prefix}Error{suffix} Min": np.min(error_values),
            f"{prefix}Error{suffix} Max": np.max(error_values),
            f"{prefix}Error{suffix} Median": np.percentile(error_values, 50),
            f"{prefix}Error{suffix} 25th Percentile": np.percentile(error_values, 25),
            f"{prefix}Error{suffix} 75th Percentile": np.percentile(error_values, 75),
            f"{prefix}Error{suffix} 90th Percentile": np.percentile(error_values, 90),
            f"{prefix}Error{suffix} 95th Percentile": np.percentile(error_values, 95),
        })
    return metrics

def compute_classification_metrics(labels, targets, prefix="inner_evaluation/"):
    """Compute classification metrics like F1-score, Recall, and Precision."""
    metrics = {}
    for suffix, suffix_load in {"": "", " Exact", : "_exact"}.items():
        metrics.update({
            f"{prefix}F1 Score{suffix}": f1_score(labels, targets[f"y_pred{suffix_load}"], average="weighted"),
            f"{prefix}Recall{suffix}": recall_score(labels, targets[f"y_pred{suffix_load}"], average="weighted"),
            f"{prefix}Precision{suffix}": precision_score(labels, targets[f"y_pred{suffix_load}"], average="weighted"),
        })
    return metrics

def compute_transitions(predictions):
    """Compute positive and negative transitions."""
    transitions = np.diff(predictions)
    return transitions == 1, transitions == -1

def compute_transition_metrics(targets, prefix="inner_evaluation/"):
    """Compute transition metrics for error analysis."""
    metrics = {}
    for suffix, suffix_load in {"": "", " Exact", : "_exact"}.items():
        pos_transitions, neg_transitions = compute_transitions(targets[f"y_correct{suffix_load}"])
        metrics.update({
            f"{prefix}Correct Transitions{suffix}": np.mean(pos_transitions),
            f"{prefix}Incorrect Transitions{suffix}": np.mean(neg_transitions),
        })
    return metrics


def get_best_alpha_from_searches(
    model_name: str,
    task_name: str,
    threshold: float = 0.05,
    method_name: str = "optimal_probe_1.0_all_layers_all_token_pos_derive_all_with_logit_only",
    alpha_path: str = "../runs/steering/df_alpha_search_results_final.csv",
):

    # df_alpha_per_combination = df_alpha.loc[df_alpha.groupby("run_name")["Metric"].idxmax(), ["run_name", "Alpha", "Type", "Reference Last", "Metric"]]
    df_alpha = pd.read_csv(alpha_path)
    df_alpha = df_alpha.loc[(df_alpha["method"].str.contains(method_name))]

    df_alpha_per_combination = df_alpha.loc[
        df_alpha[df_alpha["Type"] == "Base"].groupby("run_name")["Metric"].idxmax(),
        ["run_name", "Alpha", "Type", "Reference Last", "Metric", "model_name"],
    ]
    df_alpha_per_combination["Delta"] = (
        df_alpha_per_combination["Metric"] - df_alpha_per_combination["Reference Last"]
    )
    df_alpha_per_combination["Intervene"] = df_alpha_per_combination.apply(
        lambda row: row["Delta"]
        > np.sqrt(
            np.log(2 / threshold) / (2 * (210 if "mmlu" in row["run_name"] else 250))
        ),
        axis=1,
    )

    df_alpha_per_combination_exact = df_alpha.loc[
        df_alpha.groupby("run_name")["Metric Exact"].idxmax(),
        ["run_name", "Alpha", "Type", "Reference Exact", "Metric Exact"],
    ]
    df_alpha_per_combination_exact.rename(
        {"Alpha": "Alpha Exact", "Type": "Type Exact"}, inplace=True
    )
    df_alpha_per_combination_exact["Delta"] = (
        df_alpha_per_combination_exact["Metric Exact"]
        - df_alpha_per_combination_exact["Reference Exact"]
    )
    df_alpha_per_combination_exact["Intervene"] = df_alpha_per_combination_exact.apply(
        lambda row: row["Delta"]
        > np.sqrt(
            np.log(2 / threshold) / (2 * (210 if "mmlu" in row["run_name"] else 250))
        ),
        axis=1,
    )
    merged_df = pd.merge(
        df_alpha_per_combination,
        df_alpha_per_combination_exact,
        on="run_name",
        how="outer",
        suffixes=("_last", "_exact"),
    )

    best_alpha_last_df = merged_df.loc[
        (merged_df["model_name"] == model_name)
        & (merged_df["run_name"].str.contains(task_name))
        & (merged_df["Intervene_last"] == True),
        ["Alpha_last", "Metric"],
    ]

    best_alpha_last = (
        best_alpha_last_df.loc[:, "Alpha_last"].values[0]
        if not best_alpha_last_df.empty
        else 1.0
    )
    best_metric_last = (
        best_alpha_last_df.loc[:, "Metric"].values[0]
        if not best_alpha_last_df.empty
        else 1.0
    )

    best_alpha_exact_df = merged_df.loc[
        (merged_df["model_name"] == model_name)
        & (merged_df["run_name"].str.contains(task_name))
        & (merged_df["Intervene_exact"] == True),
        ["Alpha_exact", "Metric Exact"],
    ]
    best_alpha_exact = (
        best_alpha_exact_df.loc[:, "Alpha_exact"].values[0]
        if not best_alpha_exact_df.empty
        else None
    )
    best_metric_exact = (
        best_alpha_exact_df.loc[:, "Metric Exact"].values[0]
        if not best_alpha_exact_df.empty
        else None
    )

    return (
        best_alpha_last,
        best_alpha_exact,
        best_metric_last,
        best_metric_exact,
        merged_df,
    )


def append_metrics(
    evaluation_metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    steering_key: str,
    alpha_optimisation_target: str,
    prefix: str = "overall_evaluation/",
) -> Dict[str, Any]:

    deltas = {
        f"{prefix}Delta Accuracy": evaluation_metrics[f"{prefix}Accuracy"]
        - baseline[f"{prefix}Accuracy"],
        f"{prefix}Delta Accuracy Exact": evaluation_metrics[f"{prefix}Accuracy Exact"]
        - baseline[f"{prefix}Accuracy Exact"],
        f"{prefix}Delta F1 Score": evaluation_metrics[f"{prefix}F1 Score"]
        - baseline[f"{prefix}F1 Score"],
        f"{prefix}Delta F1 Score Exact": evaluation_metrics[f"{prefix}F1 Score Exact"]
        - baseline[f"{prefix}F1 Score Exact"],
        f"{prefix}Delta Recall": evaluation_metrics[f"{prefix}Recall"]
        - baseline[f"{prefix}Recall"],
        f"{prefix}Delta Recall Exact": evaluation_metrics[f"{prefix}Recall Exact"]
        - baseline[f"{prefix}Recall Exact"],
        f"{prefix}Delta Precision": evaluation_metrics[f"{prefix}Precision"]
        - baseline[f"{prefix}Precision"],
        f"{prefix}Delta Precision Exact": evaluation_metrics[f"{prefix}Precision Exact"]
        - baseline[f"{prefix}Precision Exact"],
        f"{prefix}Delta Error": baseline[f"{prefix}Error"]
        - evaluation_metrics[f"{prefix}Error"],
        f"{prefix}Delta Error Exact": baseline[f"{prefix}Error Exact"]
        - evaluation_metrics[f"{prefix}Error Exact"],
        f"{prefix}Corrections Total": np.sum(
            evaluation_metrics[f"{prefix}Correct Predictions"]
        )
        - np.sum(baseline[f"{prefix}Correct Predictions"]),
        f"{prefix}Corrections Percentage": (
            (
                np.sum(evaluation_metrics[f"{prefix}Correct Predictions"])
                / np.sum(baseline[f"{prefix}Correct Predictions"])
            )
            if np.sum(baseline[f"{prefix}Correct Predictions"]) != 0
            else 0.0
        ),
        f"{prefix}Corrections Total Exact": np.sum(
            evaluation_metrics[f"{prefix}Correct Predictions Exact"]
        )
        - np.sum(baseline[f"{prefix}Correct Predictions Exact"]),
        f"{prefix}Corrections Percentage Exact": (
            (
                np.sum(evaluation_metrics[f"{prefix}Correct Predictions Exact"])
                / np.sum(baseline[f"{prefix}Correct Predictions Exact"])
            )
            if np.sum(baseline[f"{prefix}Correct Predictions"]) != 0
            else 0.0
        ),
    }

    evaluation_metrics.update(deltas)

    final_pprint = f"\n\n[FINAL RESULTS] {steering_key}"
    final_pprint += (
        ""
        if alpha_optimisation_target == ""
        else f" — Evaluating {alpha_optimisation_target.capitalize()}."
    )
    print(final_pprint)
    print(
        f"[FINAL RESULTS] Delta Accuracy Last (↑): {deltas[f'{prefix}Delta Accuracy']:.3f} | Delta Error Last (↑): {deltas[f'{prefix}Delta Error']:.3f} Corrections Last Total (↑): {deltas[f'{prefix}Corrections Total']:.3f}"
    )
    print(
        f"[FINAL RESULTS] Delta Accuracy Exact (↑): {deltas[f'{prefix}Delta Accuracy Exact']:.3f} | Delta Error Exact (↑): {deltas[f'{prefix}Delta Error Exact']:.3f} Corrections Total Exact (↑): {deltas[f'{prefix}Corrections Total Exact']:.3f}\n"
    )

    return evaluation_metrics


def random_sample_activations(
    data: Dict[str, np.ndarray], k: int, seed: int = 1234
) -> Dict[str, np.ndarray]:
    """Randomly samples k elements from each array in the input dictionary using a fixed seed."""
    np.random.seed(seed)
    return {
        key: arr[np.random.choice(arr.shape[0], k, replace=False)]
        for key, arr in data.items()
    }


def random_sample_array(
    arr: np.ndarray, k: int, seed: int = 1234
) -> Dict[str, np.ndarray]:
    """Randomly samples k elements from each array in the input dictionary using a fixed seed."""
    np.random.seed(seed)
    return np.random.choice(arr, k, replace=False).tolist()


def filter_by_top_k(y_error: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Get indices for top K highest and lowest errors."""
    top_k_low_error_indices = np.argsort(y_error)[:k]
    top_k_high_error_indices = np.argsort(y_error)[-k:]
    return top_k_low_error_indices, top_k_high_error_indices


def filter_by_percentile(
    y_error: np.ndarray, lower_percentile: float = 5.0, upper_percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Get indices for low and high errors based on percentile thresholds."""
    low_threshold = np.percentile(y_error, lower_percentile)
    high_threshold = np.percentile(y_error, upper_percentile)

    low_error_indices = np.where(y_error <= low_threshold)[0]
    high_error_indices = np.where(y_error >= high_threshold)[0]
    return low_error_indices, high_error_indices


def apply_activation_filtering(
    activations_cache: dict,
    y_correct: np.ndarray,
    y_error: np.ndarray,
    filter_type: str = "top_k",
    k: int = 20,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> dict:
    """Filter activations based on the error and correctness of samples."""

    if filter_type == "top_k":
        correct_indices, incorrect_indices = filter_by_top_k(y_error, k)
    elif filter_type == "percentile":
        correct_indices, incorrect_indices = filter_by_percentile(
            y_error, lower_percentile, upper_percentile
        )
    else:
        correct_indices = np.where(y_correct == True)[0]
        incorrect_indices = np.where(y_correct == False)[0]

    # Filter activations for each layer based on selected indices!
    activations_cache_low_error = {}
    activations_cache_high_error = {}
    for layer_name in activations_cache.keys():
        layer_activations = activations_cache[layer_name]
        activations_cache_low_error[layer_name] = layer_activations[correct_indices]
        activations_cache_high_error[layer_name] = layer_activations[incorrect_indices]

    return activations_cache_low_error, activations_cache_high_error


def normalise_coeffs(a: np.ndarray, norm_mode: str = "norm") -> np.ndarray:
    """Normalise activations."""
    if norm_mode == "norm":
        norms = np.linalg.norm(a, axis=0, keepdims=True)
        norms[norms == 0] = 1e-10
        return a / norms
    elif norm_mode == "mean_std":
        return (a - np.mean(a, axis=0)) / np.std(a, axis=0)
    else:
        raise ValueError(f"Unknown normalization mode: {norm_mode}.")


def aggregate_class_logits_eta_optimiser(
    logits: torch.Tensor,
    dataset_info: Dict,
    agg_func: Callable,
    flexible_match: bool = True,
    token_pos: Optional[int] = None,
) -> torch.Tensor:
    token_ids_per_class = get_class_token_ids(dataset_info, flexible_match)
    aggregated_logits_list = []

    for token_ids in token_ids_per_class.values():
        if token_pos is not None:
            class_logits_per_class = logits[:, token_pos, token_ids]
        else:
            class_logits_per_class = logits[:, :, token_ids]

        aggregated_logits, _ = agg_func(class_logits_per_class, dim=-1)
        aggregated_logits_list.append(aggregated_logits)

    class_logits = torch.stack(aggregated_logits_list, dim=-1)
    return class_logits


def safe_serialize(value):
    """Safely serialize values for W&B table compatibility."""
    try:
        if hasattr(value, "__str__"):
            return str(value)
        if isinstance(value, list):
            if all(isinstance(v, (bool, np.bool_)) for v in value):
                return [int(v) for v in value]  # List of bools → list of ints
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
                return json.dumps(value)  # List of numbers → JSON string
            return json.dumps(value)  # General list → JSON string

        if isinstance(value, dict):
            return json.dumps(value)  # Dict → JSON string

        if isinstance(value, (np.bool_, bool)):
            return bool(value)  # Ensure proper bool type

        if isinstance(value, (np.integer, int)):
            return int(value)  # Ensure proper int type

        if isinstance(value, (np.floating, float)):
            return float(value)  # Ensure proper float type

        if isinstance(value, torch.Tensor):
            return (
                value.tolist()
                if value.numel() <= 1
                else [float(x) for x in value.flatten()]
            )

        if value is None:
            return None  # Pass None as is

    except (TypeError, ValueError) as e:
        print(
            f"[WARN] Cannot serialize value: {value} (Type: {type(value)}) - Error: {e}"
        )

    print(
        f"[WARN] Unknown serialization issue for value: {value} (Type: {type(value)})."
    )
    return None  # Default fallback
