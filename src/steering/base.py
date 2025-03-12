import json
import pickle
from typing import List, Optional, Callable, Tuple, Dict, Any, Type
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import Adam

import wandb
from contextlib import nullcontext
from scipy.special import expit, logit

from tasks.task_handler import *
from cache.cache_utils import *
from .steering_utils import *


class Steering:
    """Apply activation addition for steering toward correct predictions."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        tokenizer_kwargs: dict,
        dataset_info: dict,
        steering_kwargs: dict,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.dataset_info = dataset_info
        self.steering_kwargs = steering_kwargs

        self.eta = self.steering_kwargs.get("eta", 1.0)
        self.a = self.steering_kwargs.get("a", {})
        self.b = self.steering_kwargs.get("b", {})
        self.k = self.steering_kwargs.get("k", 0)
        self.apply_layers_to_steer = self.steering_kwargs.get(
            "apply_layers_to_steer", None
        )
        if self.apply_layers_to_steer is None:
            self.apply_layers_to_steer = list(range(len(self.model.model.layers)))
        self.apply_token_pos_to_steer = self.steering_kwargs.get(
            "apply_token_pos_to_steer", None
        )
        self.mean_centered = self.steering_kwargs.get("mean_centered", False)
        self.no_steering = self.steering_kwargs.get("no_steering", False)

        self.debug = self.steering_kwargs.get("debug", True) #FIXME

        if self.a and self.b:
            self.contrastive_vector = self.compute_diff_in_means

        self.log_with_wandb = self.steering_kwargs.get("log_with_wandb", True)
        self.init_wandb()

        self.hooks = []

    def init_wandb(self):
        """Init the wandb."""
        if self.log_with_wandb and not wandb.run:
            wandb.init(
                project="llm-error-steering",
                name=self.steering_kwargs.get("run_name", "steering"),
                config=self.steering_kwargs,
            )

    def hook_fn(self, module, input, output, layer_idx: int):
        """
        Steering hook function that dynamically handles prompt and generation phases.
        Handles 'all', 'specific', 'generation', and 'probe_position' strategies efficiently.
        """
        if self.no_steering:
            return output

        batch_size, seq_len, hidden_dim = output.shape
        assert (
            batch_size == 1
        ), "Generation mode assumes batch size is 1 during token-by-token generation."

        if self.apply_token_pos_to_steer == "all":
            return self.steer(output, layer_idx)

        elif (
            self.apply_token_pos_to_steer == "specific"
            and self.apply_token_pos_to_steer is not None
        ):
            if seq_len > self.apply_token_pos_to_steer:
                output[:, self.apply_token_pos_to_steer, :] = self.steer(
                    output[:, self.apply_token_pos_to_steer, :], layer_idx
                )
            return output

        elif self.apply_token_pos_to_steer == "generation":
            start_pos = seq_len - 1 if seq_len > 1 else 0
            output[:, start_pos:, :] = self.steer(output[:, start_pos:, :], layer_idx)
            return output

        elif (
            self.apply_token_pos_to_steer == "probe_position"
            and self.probe_match_type is not None
        ):
            if self.probe_match_type == "last":
                last_pos = seq_len - 1 if seq_len > 1 else 0
                output[:, last_pos, :] = self.steer(output[:, last_pos, :], layer_idx)
                return output
            elif self.probe_match_type == "exact":
                print(
                    "#TODO Implement apply_token_pos_to_steer == 'probe_position' and probe_match_type= 'exact'."
                )
                return output
        else:
            raise ValueError(
                f"Unsupported apply_token_pos_to_steer: {self.apply_token_pos_to_steer}"
            )

    def register_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in self.apply_layers_to_steer:

                def hook_wrapper(layer_idx):
                    def hook(module, input, output):
                        return self.hook_fn(module, input, output, layer_idx)

                    return hook

                self.hooks.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        hook_wrapper(layer_idx)
                    )
                )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    @property
    def compute_diff_in_means(self) -> Dict[int, torch.Tensor]:
        steering_vector = {}
        for layer_idx, layer in enumerate(self.a.keys()):
            a_mean = np.mean(self.a[layer], axis=0)
            b_mean = np.mean(self.b[layer], axis=0)

            if self.mean_centered:
                a_mean -= np.mean(a_mean)
                b_mean -= np.mean(b_mean)

            steering_vector[layer_idx] = torch.tensor(
                a_mean - b_mean, dtype=torch.float32
            ).to(self.model.device)
        return steering_vector

    def steer(self, activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.no_steering:
            return activations
        steering_vector = self.contrastive_vector.get(
            layer_idx, torch.zeros_like(activations).to(self.model.device)
        )
        if self.steering_kwargs.get("mean_centered", False):
            activations = activations - activations.mean(dim=0, keepdim=True)

        return activations + self.eta * steering_vector

    def preprocess_prompts(self, prompts) -> List[str]:
        return prompts

    def add_analysis(self) -> None:
        return None

    def evaluate(
        self,
        prompts,
        labels,
        #errors_baselines: Optional[Tuple[Optional[np.array], Optional[np.array]]] = None,
        grad: bool = False,
        post_process: bool = True,
        prefix="inner_evaluation/",
        alpha_calibration_token_pos_target: Optional[str] = None,
        alpha_value: Optional[int] = None,
        disable_tdqm: bool = False,
        enable_theta_tracking: Optional[bool] = None,
        model_generation_kwargs: Optional[dict] = {},
        return_projections: bool = False,
    ) -> Tuple[Dict, Dict, Dict]:
        """Run the pipeline with optional context."""
        prompts = self.preprocess_prompts(prompts)
        
        assert not (
            alpha_value is not None and alpha_calibration_token_pos_target is not None
        ), "You must provide only one of 'alpha_value' or 'alpha_calibration_token_pos_target', not both."

        if alpha_value is not None:
            self.alpha_value = alpha_value
        elif alpha_calibration_token_pos_target is not None:
            if alpha_calibration_token_pos_target == "last":
                self.alpha_value = getattr(self, "best_alpha_last")  # , 1.0
            elif alpha_calibration_token_pos_target == "exact":
                self.alpha_value = getattr(self, "best_alpha_exact")  # , 1.0
            else:
                raise ValueError(
                    f"Unsupported alpha_calibration_token_pos_target: {alpha_calibration_token_pos_target}. Expected 'last' or 'exact'."
                )

        if enable_theta_tracking:
            self.enable_theta_tracking = True
        else:
            self.enable_theta_tracking = False

        with self:
            completions = generate_completions(
                model=self.model,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                prompts=prompts,
                batch_size=1,
                device=self.model.device,
                flexible_match=True,
                dataset_info=self.dataset_info,
                save_dir=f"../runs/",
                save_key="",
                save=False,
                grad=grad,
                use_cache=True,
                disable_tdqm=disable_tdqm,
                model_generation_kwargs=model_generation_kwargs,
            )
            clean_gpus()
            if return_projections:
                return (
                    self.projections,
                    completions["completion_sequence_lengths"],
                    completions["match_indices"],
                )

            logits, softmaxs = compute_logits(
                model=self.model,
                completions=completions["completions"],
                mode="all",
                flexible_match=True,
                position=None,
                dataset_info=self.dataset_info,
                save_dir=f"../runs/",
                save_key="",
                save=False,
                grad=grad,
                use_cache=False,
                disable_tdqm=disable_tdqm,
            )
            clean_gpus()
            targets = compute_targets(
                y_softmax_all=softmaxs,
                y_true=labels,
                dataset_info=self.dataset_info,
                prompt_sequence_lengths=completions["prompt_sequence_lengths"],
                match_indices=completions["match_indices"],
                save_dir=f"../runs/",
                save_key="",
                save=False,
                grad=grad,
            )
            clean_gpus()

        if post_process:
            prefix = f"{prefix}"
            
            # Append all the metrics!
            evaluation_metrics = {}
            evaluation_metrics.update(compute_error_metrics(targets, prefix))
            evaluation_metrics.update(compute_classification_metrics(labels, targets, prefix))
            evaluation_metrics.update(compute_transition_metrics(targets, prefix))
            
            if log_with_wandb:
                wandb.log(evaluation_metrics)
                
            self.add_analysis()

            return evaluation_metrics

        return targets

    def apply_mean_magnitude_scaling(
        self, vector: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Adjust the probe coefficients using global mean L2 magnitude of activations.
        """
        activation_magnitude = torch.norm(activations, p=2, dim=(0, 1))
        vector_magnitude = torch.norm(vector, p=2)
        scaling_factor = activation_magnitude / (vector_magnitude + 1e-8)
        if self.debug:
            print(f"Scaling Factor (Layer {layer_idx}): {scaling_factor.item()}")
        return vector * scaling_factor