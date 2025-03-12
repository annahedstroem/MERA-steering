from typing import List, Optional, Callable, Tuple, Dict, Any, Type
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks.task_handler import *
from cache.cache_utils import *
from .steering_utils import *
from .base import Steering


class SteeringByPrompt(Steering):
    """Adjust activations adding in-context prompt addition."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        tokenizer_kwargs: dict,
        dataset_info: dict,
        steering_kwargs: dict,
    ):
        super().__init__(
            model, tokenizer, tokenizer_kwargs, dataset_info, steering_kwargs
        )
        self.prompt_addition = self.steering_kwargs.get(
            "prompt_addition", "Think before you answer."
        )

    def preprocess_prompts(self, prompts) -> List[str]:
        return [
            p.replace("Answer:", f"{self.prompt_addition} Answer:") for p in prompts
        ]
