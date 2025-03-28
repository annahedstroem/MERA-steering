from typing import List, Optional, Callable, Tuple, Dict, Any, Type
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from contextlib import nullcontext
from scipy.special import expit, logit
from sklearn.metrics import f1_score, recall_score, precision_score

from tasks.task_handler import *
from cache.cache_utils import *
from .steering_utils import *
from .base import Steering


class SteeringByProbe(Steering):
    """Adjust activations using linear probe coefficients in the sparse space."""

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
        assert (
            "probe_weights" in self.steering_kwargs
        ), "Pass 'probe_weights' parameter (dict) to run SteeringByProbe class."
        assert (
            "mode" in self.steering_kwargs
        ), "Pass 'mode' parameter (str) to run SteeringByProbe class."
        assert self.steering_kwargs["mode"] in [
            "optimal_probe",
            "optimal_contrastive",
            "multiplicative_probe",
            "additive_probe",
            "internal_projection",
        ]

        self.probe_weights = {
            layer_idx: torch.tensor(layer_coeffs, dtype=torch.float32).to(
                self.model.device
            )
            for layer_idx, layer_coeffs in self.steering_kwargs["probe_weights"].items()
        }
        self.mode = self.steering_kwargs.get("mode")

        self.lmbda = self.steering_kwargs.get("lmbda", 1.0)
        self.alpha_value = self.steering_kwargs.get("alpha_value", None)
        self.derive_with_all = self.steering_kwargs.get("derive_with_all", True)
        self.derive_with_sigmoid = self.steering_kwargs.get(
            "derive_with_sigmoid", False
        )
        self.derive_with_logit = self.steering_kwargs.get("derive_with_logit", True)
        self.probe_vector = self.probe_weights
        self.internal_projections = {
            layer_idx: [] for layer_idx in self.steering_kwargs["probe_weights"]
        }

        self.normalise_coeffs = self.steering_kwargs.get("normalise_coeffs", False)

        if self.apply_layers_to_steer is None:
            self.apply_layers_to_steer = list(self.probe_weights.keys())

        self.logging_theta_table_key = self.steering_kwargs.get(
            "logging_theta_table_key", "table_name"
        )
        self.enable_theta_tracking = self.steering_kwargs.get(
            "enable_theta_tracking", False
        )

        self.internal_projection_with_probe = self.steering_kwargs.get(
            "internal_projection_with_probe", False
        )
        self.internal_projection_with_contrastive = self.steering_kwargs.get(
            "internal_projection_with_contrastive", False
        )

        if self.debug:
            print("Layers to steer:", self.apply_layers_to_steer)

    def derive_closed_form_vector(
        self,
        activations: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:

        assert (
            self.alpha_value is not None
        ), "'alpha_value' cannot be None in 'derive_closed_form_vector' func."

        # Compute the dot product per token position (batch_size, token_positions).
        wTx = torch.matmul(
            activations.to(self.model.device), vector.to(self.model.device)
        )
        wTx_transformed = wTx
        if self.derive_with_sigmoid:
            wTx_transformed = torch.special.expit(wTx)

        alpha_transformed = self.alpha_value
        if self.derive_with_logit:
            alpha_transformed = torch.special.logit(
                torch.tensor(
                    self.alpha_value, dtype=torch.float32, device=self.model.device
                )
            )

        # Check if condition is true per token position (batch_size, token_positions).
        condition = wTx_transformed > alpha_transformed

        # Derive the optimal value.
        theta = (
            (alpha_transformed - wTx_transformed) / torch.norm(vector, p=2) ** 2 + 1e-8
        ).unsqueeze(-1) * vector.unsqueeze(0).unsqueeze(0)

        # if self.debug:
        #    print(
        #        f"[DEBUG] In 'derive_closed_form_vector' — Using alpha_value {self.alpha_value} \
        #        | derive_with_sigmoid {self.derive_with_sigmoid} \
        #        | derive_with_logit {self.derive_with_logit} \
        #        | derive_with_all {self.derive_with_all}"
        #    )

        # Return the value for all token positions or the last including the generation.
        if self.derive_with_all:
            optimal_theta = torch.where(
                condition.unsqueeze(-1).to(self.model.device),
                theta.to(self.model.device),
                torch.zeros_like(activations).to(self.model.device),
            ).to(self.model.device)
        else:
            optimal_theta = torch.where(
                condition[:, -1].unsqueeze(-1).to(self.model.device),
                theta[:, -1, :].to(self.model.device),
                torch.zeros_like(vector).to(self.model.device),
            ).to(self.model.device)

        return optimal_theta, theta, condition

    def steer(self, activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Main functionality that steerst the model on a token position and layer basis."""
        if self.normalise_coeffs:
            activations = self.apply_mean_magnitude_scaling(
                probe_weights=self.probe_weights[layer_idx], activations=activations
            )

        if self.mode == "multiplicative_probe":
            return activations * self.lmbda * -self.probe_weights[layer_idx]

        elif self.mode == "additive_probe":
            return activations + self.lmbda * -self.probe_weights[layer_idx]

        print(
            "[DEBUG] Returning unsteered activations, as no 'mode' matched the implementation."
        )
        return activations
