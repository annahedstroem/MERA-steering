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
    "inner_evaluation/Delta Accuracy Last",
    "inner_evaluation/Delta Accuracy Exact",
    "inner_evaluation/Delta F1 Score Last",
    "inner_evaluation/Delta F1 Score Exact",
    "inner_evaluation/Delta Recall Last",
    "inner_evaluation/Delta Recall Exact",
    "inner_evaluation/Delta Precision Last",
    "inner_evaluation/Delta Precision Exact",
    "inner_evaluation/Delta Error Last",
    "inner_evaluation/Delta Error Exact",
    "inner_evaluation/Corrections Total Last",
    "inner_evaluation/Corrections Percentage Last",
    "inner_evaluation/Corrections Total Exact",
    "inner_evaluation/Corrections Percentage Exact",
    "inner_evaluation/Transitions (0->1) Last",
    "inner_evaluation/Transitions (0->1) Exact",
    "inner_evaluation/Transitions (0->0) Last",
    "inner_evaluation/Transitions (0->0) Exact",
    "inner_evaluation/Transitions (1->1) Last",
    "inner_evaluation/Transitions (1->1) Exact",
    "inner_evaluation/Transitions (1->0) Last",
    "inner_evaluation/Transitions (1->0) Exact",
]


def compute_spi(unsteered_acc: float, delta_acc: float):
    if delta_acc > 0:
        return delta_acc / (1 - unsteered_acc)
    elif delta_acc < 0:
        return delta_acc / (unsteered_acc + 1e-10)
    elif delta_acc == 0:
        return 0


def compute_error_metrics(targets, prefix="inner_evaluation/"):
    """Compute error-related metrics."""
    metrics = {}
    for suffix, suffix_load in {" Last": "", " Exact": "_exact"}.items():
        acc_values = targets[f"y_correct{suffix_load}"]
        error_values = 1 - np.array(targets[f"y_softmax{suffix_load}"])
        metrics.update(
            {
                f"{prefix}Error{suffix}": np.mean(error_values),
                f"{prefix}Accuracy{suffix}": np.mean(acc_values),
                f"{prefix}Error{suffix} Std": np.std(error_values),
                f"{prefix}Accuracy{suffix} Std": np.std(acc_values),
                f"{prefix}Error{suffix} Min": np.min(error_values),
                f"{prefix}Error{suffix} Max": np.max(error_values),
                f"{prefix}Error{suffix} Median": np.percentile(error_values, 50),
                f"{prefix}Error{suffix} 25th Percentile": np.percentile(
                    error_values, 25
                ),
                f"{prefix}Error{suffix} 75th Percentile": np.percentile(
                    error_values, 75
                ),
                f"{prefix}Error{suffix} 90th Percentile": np.percentile(
                    error_values, 90
                ),
                f"{prefix}Error{suffix} 95th Percentile": np.percentile(
                    error_values, 95
                ),
                f"{prefix}Correct Predictions{suffix}": acc_values,
            }
        )
    return metrics


def compute_classification_metrics(labels, targets, prefix="inner_evaluation/"):
    """Compute classification metrics like F1-score, Recall, and Precision."""
    metrics = {}
    for suffix, suffix_load in {" Last": "", " Exact": "_exact"}.items():
        metrics.update(
            {
                f"{prefix}F1 Score{suffix}": f1_score(
                    labels, targets[f"y_pred{suffix_load}"], average="weighted"
                ),
                f"{prefix}Recall{suffix}": recall_score(
                    labels, targets[f"y_pred{suffix_load}"], average="weighted"
                ),
                f"{prefix}Precision{suffix}": precision_score(
                    labels, targets[f"y_pred{suffix_load}"], average="weighted"
                ),
            }
        )
    return metrics


def compute_transitions(
    baseline: np.ndarray, steered: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all transitions from baseline to steered: 0→1, 1→0, 0→0, 1→1."""
    baseline = np.array(baseline)
    steered = np.array(steered)
    return (
        (baseline == 0) & (steered == 1),  # Recovery!
        (baseline == 1) & (steered == 0),  # Regression!
        (baseline == 0) & (steered == 0),  # Persistent Error!
        (baseline == 1) & (steered == 1),  # Stability!
    )


def compute_transition_metrics(
    baseline_correct: Dict[str, np.ndarray],
    steered_correct: Dict[str, np.ndarray],
    prefix: str = "inner_evaluation/",
    suffix: str = "",
) -> Dict[str, float]:
    """Compute transition metrics from baseline to steered."""
    metrics = {}
    t_0to1, t_1to0, t_0to0, t_1to1 = compute_transitions(
        baseline_correct, steered_correct
    )
    metrics.update(
        {
            f"{prefix}Transitions (0->1){suffix}": np.mean(t_0to1),
            f"{prefix}Transitions (1->0){suffix}": np.mean(t_1to0),
            f"{prefix}Transitions (0->0){suffix}": np.mean(t_0to0),
            f"{prefix}Transitions (1->1){suffix}": np.mean(t_1to1),
        }
    )
    return metrics


def append_metrics(
    evaluation_metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    steering_key: str,
    alpha_optimisation_target: str,
    prefix: str = "overall_evaluation/",
) -> Dict[str, Any]:

    deltas = {
        f"{prefix}Delta Accuracy Last": evaluation_metrics[f"{prefix}Accuracy Last"]
        - baseline[f"{prefix}Accuracy Last"],
        f"{prefix}Delta Accuracy Exact": evaluation_metrics[f"{prefix}Accuracy Exact"]
        - baseline[f"{prefix}Accuracy Exact"],
        f"{prefix}Delta F1 Score Last": evaluation_metrics[f"{prefix}F1 Score Last"]
        - baseline[f"{prefix}F1 Score Last"],
        f"{prefix}Delta F1 Score Exact": evaluation_metrics[f"{prefix}F1 Score Exact"]
        - baseline[f"{prefix}F1 Score Exact"],
        f"{prefix}Delta Recall Last": evaluation_metrics[f"{prefix}Recall Last"]
        - baseline[f"{prefix}Recall Last"],
        f"{prefix}Delta Recall Exact": evaluation_metrics[f"{prefix}Recall Exact"]
        - baseline[f"{prefix}Recall Exact"],
        f"{prefix}Delta Precision Last": evaluation_metrics[f"{prefix}Precision Last"]
        - baseline[f"{prefix}Precision Last"],
        f"{prefix}Delta Precision Exact": evaluation_metrics[f"{prefix}Precision Exact"]
        - baseline[f"{prefix}Precision Exact"],
        f"{prefix}Delta Error Last": baseline[f"{prefix}Error Last"]
        - evaluation_metrics[f"{prefix}Error Last"],
        f"{prefix}Delta Error Exact": baseline[f"{prefix}Error Exact"]
        - evaluation_metrics[f"{prefix}Error Exact"],
        f"{prefix}Corrections Total Last": np.sum(
            evaluation_metrics[f"{prefix}Correct Predictions Last"]
        )
        - np.sum(baseline[f"{prefix}Correct Predictions Last"]),
        f"{prefix}Corrections Percentage Last": (
            (
                np.sum(evaluation_metrics[f"{prefix}Correct Predictions Last"])
                / np.sum(baseline[f"{prefix}Correct Predictions Last"])
            )
            if np.sum(baseline[f"{prefix}Correct Predictions Last"]) != 0
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
            if np.sum(baseline[f"{prefix}Correct Predictions Exact"]) != 0
            else 0.0
        ),
    }
    deltas[f"{prefix}SPI Last"] = compute_spi(
        unsteered_acc=baseline[f"{prefix}Accuracy Last"],
        delta_acc=deltas[f"{prefix}Delta Accuracy Last"],
    )
    deltas[f"{prefix}SPI Exact"] = compute_spi(
        unsteered_acc=baseline[f"{prefix}Accuracy Exact"],
        delta_acc=deltas[f"{prefix}Delta Accuracy Exact"],
    )

    transitions = compute_transition_metrics(
        baseline[f"{prefix}Correct Predictions Last"],
        evaluation_metrics[f"{prefix}Correct Predictions Last"],
        suffix=" Last",
    )
    transitions_exact = compute_transition_metrics(
        baseline[f"{prefix}Correct Predictions Exact"],
        evaluation_metrics[f"{prefix}Correct Predictions Exact"],
        suffix=" Exact",
    )
    update_metrics = {**deltas, **transitions, **transitions_exact}
    evaluation_metrics.update(update_metrics)

    final_pprint = f"\n\n[FINAL RESULTS] {steering_key}"
    final_pprint += (
        ""
        if alpha_optimisation_target == ""
        else f" — Evaluating {alpha_optimisation_target.capitalize()}."
    )
    print(final_pprint)
    print(
        f"[FINAL RESULTS] Last — SPI (↑): {update_metrics[f'{prefix}SPI Last']} | Delta Accuracy (↑): {deltas[f'{prefix}Delta Accuracy Last']:.3f} | Delta Error (↑): {deltas[f'{prefix}Delta Error Last']:.3f} Corrections Total (↑): {deltas[f'{prefix}Corrections Total Last']:.3f}"
    )
    print(
        f"[FINAL RESULTS] Exact — SPI (↑): {update_metrics[f'{prefix}SPI Exact']} | Delta Accuracy (↑): {deltas[f'{prefix}Delta Accuracy Exact']:.3f} | Delta Error (↑): {deltas[f'{prefix}Delta Error Exact']:.3f} Corrections Total (↑): {deltas[f'{prefix}Corrections Total Exact']:.3f}\n"
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
                return [int(v) for v in value]  # List of bools → list of ints!
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
                return json.dumps(value)  # List of numbers → JSON string!
            return json.dumps(value)  # General list → JSON string!

        if isinstance(value, dict):
            return json.dumps(value)  # Dict → JSON string!

        if isinstance(value, (np.bool_, bool)):
            return bool(value)  # Ensure proper bool type!

        if isinstance(value, (np.integer, int)):
            return int(value)  # Ensure proper int type!

        if isinstance(value, (np.floating, float)):
            return float(value)  # Ensure proper float type!

        if isinstance(value, torch.Tensor):
            return (
                value.tolist()
                if value.numel() <= 1
                else [float(x) for x in value.flatten()]
            )

        if value is None:
            return None  # Pass None as is!

    except (TypeError, ValueError) as e:
        print(
            f"[WARN] Cannot serialize value: {value} (Type: {type(value)}) - Error: {e}"
        )

    print(
        f"[WARN] Unknown serialization issue for value: {value} (Type: {type(value)})."
    )
    return None  # Default fallback
