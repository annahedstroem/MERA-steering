from typing import List, Optional, Callable, Tuple, Dict, Any, Type
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from scipy.special import expit, logit
from sklearn.metrics import f1_score, recall_score, precision_score

from tasks.task_handler import *
from cache.cache_utils import *
from .steering_utils import *
from .by_probe import SteeringByProbe


class MERA(SteeringByProbe):
    """
    Implementation of MERA with calibration of the steering threshold (alpha).

    For details regarding hyperparameters see SteeringProbe and Steering base classes!
    """

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
        self.METRIC_KEYS = ["Accuracy", "F1 Score", "Recall", "Precision", "Error"]
        self.TOKEN_POS = ["Last", "Exact"]
        self.METRIC_KEYS_FULL = [
            f"{k} {p}" for k in self.METRIC_KEYS for p in self.TOKEN_POS
        ]

        # Alpha grid search parameters.
        self.prefix = self.steering_kwargs.get("prefix", "inner_evaluation/")
        self.logging_calibration_table_key = self.steering_kwargs.get(
            "logging_calibration_table_key"
        )
        self.alpha_range = self.steering_kwargs.get(
            "alpha_range", np.linspace(0.1, 0.9, 9)
        )
        self.nr_samples = self.steering_kwargs.get("nr_samples", 250)
        self.enable_constraint = self.steering_kwargs.get("enable_constraint", True)
        self.constraint_value = self.steering_kwargs.get("constraint_value", 2)
        self.objective_key = self.steering_kwargs.get("objective_key", f"Accuracy")
        self.objective_key = f"{self.prefix}{self.objective_key}"
        self.objective_key_exact = self.objective_key + " Exact"
        self.ref_prompts = self.steering_kwargs.get("ref_prompts", [])
        self.ref_labels = self.steering_kwargs.get("ref_labels", [])

        self.best_alpha_last = self.steering_kwargs.get("best_alpha_last", None)
        self.best_alpha_exact = self.steering_kwargs.get("best_alpha_exact", None)
        self.best_metric_last = self.steering_kwargs.get("best_metric_last", None)
        self.best_metric_exact = self.steering_kwargs.get("best_metric_exact", None)

        # FIXME LATER.
        if self.best_alpha_last is None or self.best_alpha_exact is None:
            print("\n[INFO] Calibrating alpha...")
            self.calibrate_alpha()
        else:
            print(
                "\n[SKIPPING GRID SEARCH] Using precomputed best_alpha. Skipping redundant evaluation."
            )

    def steer(self, activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Main functionality that steerst the model on a token position and layer basis."""
        if self.normalise_coeffs:
            activations = self.apply_mean_magnitude_scaling(
                probe_weights=self.probe_weights[layer_idx], activations=activations
            )

        if self.mode == "optimal_probe":
            optimal_theta, theta, condition = self.optimise_steering_closed_form(
                activations, self.probe_weights[layer_idx]
            )
            return activations.to(self.model.device) + optimal_theta.to(
                self.model.device
            )

        elif self.mode == "optimal_contrastive":
            optimal_theta, theta, condition = self.optimise_steering_closed_form(
                activations, self.contrastive_vector[layer_idx]
            )
            return activations.to(self.model.device) + optimal_theta.to(
                self.model.device
            )

        elif (
            self.mode == "internal_projection"
        ):  # used for token position analysis in the paper
            if self.internal_projection_with_contrastive:
                steering_vector = self.contrastive_vector.get(
                    layer_idx, torch.zeros_like(activations).to(self.model.device)
                )
                self.internal_projections[layer_idx].append(
                    torch.matmul(activations.to(self.model.device), steering_vector)
                )

            elif self.internal_projection_with_probe:
                probe_weights = self.probe_weights.get(
                    layer_idx, torch.zeros_like(activations).to(self.model.device)
                )
                self.internal_projections[layer_idx].append(
                    torch.matmul(activations.to(self.model.device), probe_weights)
                )

        print(
            "[DEBUG] Returning unsteered activations, as no 'mode' matched the implementation."
        )
        return activations

    def optimise_steering_closed_form(
        self,
        activations: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:

        assert (
            self.alpha_value is not None
        ), "'alpha_value' cannot be None in 'optimise_steering_closed_form' func."

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
        #        f"[DEBUG] In 'optimise_steering_closed_form' — Using alpha_value {self.alpha_value} \
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

    def create_alpha_results_table(self) -> wandb.Table:
        """Create wandb Table with all metric columns."""
        columns = (
            ["Alpha", "Type", "Best Alpha Last", "Best Alpha Exact"]
            + [f"Reference {k}" for k in self.METRIC_KEYS_FULL]
            + [f"Current {k}" for k in self.METRIC_KEYS_FULL]
            + [f"Delta {k}" for k in self.METRIC_KEYS_FULL]
        )
        return wandb.Table(columns=columns)

    def compute_metric_deltas(self, current: dict, reference: dict) -> Dict[str, float]:
        """Compute metric deltas from current and reference values."""
        deltas = {}
        for key in self.METRIC_KEYS_FULL:
            cur = current.get(f"{self.prefix}{key}", None)
            ref = reference.get(f"{self.prefix}{key}", None)
            if cur is None or ref is None:
                deltas[key] = None
            else:
                # Maximize all but Error.
                deltas[key] = cur - ref if key != "Error" else ref - cur
        return deltas

    def log_metrics_to_wandb_table(
        self,
        table: wandb.Table,
        alpha: float,
        search_type: str,
        ref: dict,
        current: dict,
        deltas: dict,
        best_alpha_last: float,
        best_alpha_exact: float,
    ) -> None:
        """Add one row of metrics to wandb table."""
        row = [alpha, search_type, best_alpha_last, best_alpha_exact]
        for key in self.METRIC_KEYS_FULL:
            row.append(ref.get(f"{self.prefix}{key}", None))
        for key in self.METRIC_KEYS_FULL:
            row.append(current.get(f"{self.prefix}{key}", None))
        for key in self.METRIC_KEYS_FULL:
            row.append(deltas.get(key, None))
        table.add_data(*row)

    def compute_and_log_metrics(
        self,
        alpha: float,
        ref_metrics: dict,
        current_metrics: dict,
        alpha_results_table: wandb.Table,
        search_type: str,
        best_alpha_last: float,
        best_alpha_exact: float,
    ) -> Dict[str, float]:
        """Compute deltas and log all evaluation metrics into WandB."""
        print(f"[INFO] Logging alpha={alpha:.3f} metrics to WandB table.")

        deltas = self.compute_metric_deltas(current_metrics, ref_metrics)
        self.log_metrics_to_wandb_table(
            table=alpha_results_table,
            alpha=alpha,
            search_type=search_type,
            ref=ref_metrics,
            current=current_metrics,
            deltas=deltas,
            best_alpha_last=best_alpha_last,
            best_alpha_exact=best_alpha_exact,
        )

        return deltas

    def perform_calibration(
        self,
        alpha_ranges: List[float],
        ref_metrics: dict,
        best_metrics: Dict[str, float],
        best_alphas: Dict[str, float],
        improved: Dict[str, bool],
        alpha_results_table: wandb.Table,
        search_type: str,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, bool]]:
        """Perform alpha grid search and log metrics."""
        for alpha in alpha_ranges:

            print(f"[INFO] Evaluating alpha_value: {alpha:.3f}")
            current_metrics = self.evaluate(
                prompts=self.ref_prompts,
                labels=self.ref_labels,
                alpha_value=alpha,
                prefix="inner_evaluation/",
            )

            for pos in self.TOKEN_POS:
                for metric_key in self.METRIC_KEYS:
                    full_key = f"{self.prefix}{metric_key} {pos}"
                    current_val = current_metrics[full_key]
                    if (
                        metric_key == "Error"
                        and current_val <= best_metrics[pos][metric_key]
                    ):
                        best_metrics[pos][metric_key] = current_val
                        best_alphas[pos][metric_key] = alpha
                        improved[pos][metric_key] = True
                    elif (
                        metric_key != "Error"
                        and current_val >= best_metrics[pos][metric_key]
                    ):
                        best_metrics[pos][metric_key] = current_val
                        best_alphas[pos][metric_key] = alpha
                        improved[pos][metric_key] = True
                    print(
                        f"[DEBUG] {metric_key} {pos} | Current: {current_val:.3f} | Best: {best_metrics[pos][metric_key]:.3f}"
                    )

            # Select which alphas to use during actual steering!
            key = self.objective_key.replace(self.prefix, "")
            best_alpha_last = best_alphas["Last"][key]
            best_alpha_exact = best_alphas["Exact"][key]

            deltas = self.compute_and_log_metrics(
                alpha,
                ref_metrics,
                current_metrics,
                alpha_results_table,
                search_type,
                best_alpha_last,
                best_alpha_exact,
            )

            print(f"[INFO] Best metrics so far: {best_metrics}")
            print(f"[INFO] Best alphas so far: {best_alpha_last} | {best_alpha_exact}")

        return best_alphas, best_metrics, improved

    def set_alpha_attrs(
        self,
        improved: bool,
        ref_value: float,
        best_alpha: float,
        best_metric: float,
        alpha_calibration_token_pos_target: str,
    ):
        """Set class attributes used for offline, evaluation mode steering."""
        if not improved:
            print(
                f"[CALIBRATION] No improvement detected during alpha search for {alpha_calibration_token_pos_target.upper()} mode. Defaulting to alpha_{alpha_calibration_token_pos_target} = 1.0 (No intervention)."
            )
            setattr(self, f"best_alpha_{alpha_calibration_token_pos_target}", 1.0)
            setattr(
                self, f"best_metric_{alpha_calibration_token_pos_target}", ref_value
            )
        else:
            setattr(
                self, f"best_alpha_{alpha_calibration_token_pos_target}", best_alpha
            )
            setattr(
                self, f"best_metric_{alpha_calibration_token_pos_target}", best_metric
            )
            self.steering_kwargs[f"best_alpha_{alpha_calibration_token_pos_target}"] = (
                best_alpha
            )
            self.steering_kwargs[
                f"best_metric_{alpha_calibration_token_pos_target}"
            ] = best_metric

    def calibrate_alpha(self) -> None:
        """Main entry to trigger alpha calibration."""
        ref_metrics = self.evaluate(
            prompts=self.ref_prompts,
            labels=self.ref_labels,
            alpha_value=1.0,
            prefix=self.prefix,
        )
        best_metrics = {
            pos: {k: ref_metrics[f"{self.prefix}{k} {pos}"] for k in self.METRIC_KEYS}
            for pos in self.TOKEN_POS
        }
        best_alphas = {
            pos: {k: 1.0 for k in self.METRIC_KEYS} for pos in self.TOKEN_POS
        }
        improved = {pos: {k: False for k in self.METRIC_KEYS} for pos in self.TOKEN_POS}

        alpha_results_table = self.create_alpha_results_table()

        best_alphas, best_metrics, improved = self.perform_calibration(
            alpha_ranges=self.alpha_range,
            ref_metrics=ref_metrics,
            best_metrics=best_metrics,
            best_alphas=best_alphas,
            improved=improved,
            alpha_results_table=alpha_results_table,
            search_type="Base",
        )

        if self.log_with_wandb:
            wandb.log(
                {
                    f"alpha_search_results_table/{self.logging_calibration_table_key}": alpha_results_table
                }
            )
            print(
                f"[DEBUG] Logged table under alpha_search_results_table/{self.logging_calibration_table_key}"
            )

        self.best_alpha_results = best_alphas
        self.best_metric_results = best_metrics

        key = self.objective_key.replace(self.prefix, "")
        self.set_alpha_attrs(
            improved=improved["Last"][key],
            ref_value=ref_metrics[f"{self.objective_key} Last"],
            best_alpha=self.best_alpha_results["Last"][key],
            best_metric=self.best_metric_results["Last"][key],
            alpha_calibration_token_pos_target="last",
        )
        self.set_alpha_attrs(
            improved=improved["Exact"][key],
            ref_value=ref_metrics[f"{self.objective_key_exact}"],
            best_alpha=self.best_alpha_results["Exact"][key],
            best_metric=self.best_metric_results["Exact"][key],
            alpha_calibration_token_pos_target="exact",
        )

    def get_alpha_results(self) -> List[Tuple[float, float]]:
        """Retrieve the results from the alpha grid search."""
        if hasattr(self, "best_alpha_results"):
            return list(self.best_alpha_results.items())
        else:
            raise ValueError(
                "No alpha search results found. Run calibration_alpha() first."
            )
