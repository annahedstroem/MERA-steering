"""This module includes the supported steering methods, tasks and models."""

from typing import List, Optional, Callable, Tuple, Dict, Any, Type
from .base import Steering
from .by_prompt import SteeringByPrompt
from .by_probe import SteeringByProbe
from .by_mera import MERA

SUPPORTED_METHODS = {
    "No steering": "no_steering",
    "With Mera (error)": "optimal_probe",
    "With Mera (logistic)": "optimal_logistic_probe",
    "With Mera (contrastive)": "optimal_contrastive",
    "With Mera (worst)": "sub_optimal_probe",
    "With Mera (median)": "median_optimal_probe",
    "Baseline (error)": "additive_probe",
    "Baseline (worst)": "additive_sub_probe",
    "Baseline (median)": "additive_median_probe",
    "Baseline (logistic)": "additive_logistic_probe",
    "Baseline (contrastive)": "vanilla_contrastive",
    "Baseline (prompting)": "prompt_steering",
}

SUPPORTED_TASKS = [
    "sentiment_analysis",
    "yes_no_question",
    "mmlu_high_school",
    "mmlu_professional",
    "sms_spam",
]

SUPPORTED_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-3B",
]


def init_steering(steering_kwargs: Dict) -> Type[Steering]:
    """Select the appropriate Steering class based on steering_kwargs."""
    # the order is important
    if "alpha_range" in steering_kwargs:
        selected_class = MERA
    elif "probe_weights" in steering_kwargs:
        selected_class = SteeringByProbe
    elif "prompt_addition" in steering_kwargs:
        selected_class = SteeringByPrompt
    else:
        selected_class = Steering
    print(f"[INFO] Selected Steering Class: {selected_class.__name__}.")
    return selected_class
