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
        self.objective_key = self.steering_kwargs.get(
            "objective_key", f"{self.prefix}Accuracy"
        )
        self.objective_key_exact = self.steering_kwargs.get(
            "objective_key_exact", self.objective_key + " Exact"
        )
        self.refine_best_alpha = self.steering_kwargs.get("refine_best_alpha", False)
        self.ref_prompts = self.steering_kwargs.get("ref_prompts", [])
        self.ref_labels = self.steering_kwargs.get("ref_labels", [])

        self.best_alpha_last = self.steering_kwargs.get("best_alpha_last", None)
        self.best_alpha_exact = self.steering_kwargs.get("best_alpha_exact", None)
        self.best_metric_last = self.steering_kwargs.get("best_metric_last", None)
        self.best_metric_exact = self.steering_kwargs.get("best_metric_exact", None)

        # FIXME LATER.
        if (
            self.refine_best_alpha
            or self.best_alpha_last is None
            or self.best_alpha_exact is None
        ):
            print("\n[CALIBRATION] Searching for alpha...")
            self.calibration_alpha()
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
            optimal_theta, theta, condition = self.derive_closed_form_vector(
                activations, self.probe_weights[layer_idx]
            )  # , apply_sigmoid=True)
            if self.enable_theta_tracking:
                self.update_theta_statistics(optimal_theta, theta, condition, layer_idx)
            return activations.to(self.model.device) + optimal_theta.to(
                self.model.device
            )

        elif self.mode == "optimal_contrastive":
            optimal_theta, theta, condition = self.derive_closed_form_vector(
                activations, self.contrastive_vector[layer_idx]
            )
            if self.enable_theta_tracking:
                self.update_theta_statistics(
                    optimal_theta, theta, condition, layer_idx, activations.shape
                )
            return activations.to(self.model.device) + optimal_theta.to(
                self.model.device
            )

        elif self.mode == "internal_projection":
            if self.internal_projection_with_contrastive:
                steering_vector = self.contrastive_vector.get(
                    layer_idx, torch.zeros_like(activations).to(self.model.device)
                )
                self.internal_projections[layer_idx].append(
                    torch.matmul(activations.to(self.model.device), steering_vector)
                )

            elif self.internal_projection_with_probe:
                probe_vector = self.probe_vector.get(
                    layer_idx, torch.zeros_like(activations).to(self.model.device)
                )
                self.internal_projections[layer_idx].append(
                    torch.matmul(activations.to(self.model.device), probe_vector)
                )

        print(
            "[DEBUG] Returning unsteered activations, as no 'mode' matched the implementation."
        )
        return activations

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

    def calibration_alpha(self) -> None:
        """
        Perform grid search over alpha to calibrate the value the objective.
        """

        if (
            self.refine_best_alpha
            or self.best_alpha_last is None
            or self.best_alpha_exact is None
        ):

            assert self.objective_key in [
                "Accuracy",
                "F1 Score",
                "Recall",
                "Precision",
                "Error",
                "Error Exact",
                "Accuracy Exact",
                "F1 Score Exact",
                "Recall Exact",
                "Precision Exact",
            ], f"Invalid objective_key: {self.objective_key}"

            # Step 1. Establish reference baseline of the objective!
            print("[INFO] Evaluating reference performance without steering...")
            ref_metrics = self.evaluate(
                prompts=self.ref_prompts,
                labels=self.ref_labels,
                alpha_value=1.0,
                prefix="inner_evaluation/",
            )
            ref_metric_value_last = ref_metrics.get(
                f"{self.prefix}{self.objective_key}", None
            )
            ref_metric_value_exact = ref_metrics.get(
                f"{self.prefix}{self.objective_key_exact}", None
            )
            ref_y_correct = ref_metrics[f"{self.prefix}Correct Predictions"]
            ref_y_correct_exact = ref_metrics[f"{self.prefix}Correct Predictions Exact"]

            if ref_metric_value_last is None or ref_metric_value_exact is None:
                raise ValueError(
                    f"Objective keys '{self.objective_key}' or '{self.objective_key_exact}' not found in reference metrics."
                )
            print(
                f"[CALIBRATION] Reference Last {self.objective_key}: {ref_metric_value_last:.4f}"
            )
            print(
                f"[CALIBRATION] Reference Exact {self.objective_key_exact}: {ref_metric_value_exact:.4f}"
            )

            # Step 2. Calibrate alpha.
            master_metrics = {}
            alpha_results_table = wandb.Table(
                columns=[
                    "Alpha",
                    "Type",
                    "Reference Last",
                    "Metric",
                    "Delta",
                    "Corrections Total Last",
                    "Corrections Percentage Last",
                    "Reference Exact",
                    "Metric Exact",
                    "Delta Exact",
                    "Corrections Total Exact",
                    "Corrections Percentage Exact",
                ]
            )

        if self.best_alpha_last is None or self.best_alpha_exact is None:

            print("\n[CALIBRATION] Searching for alpha...")
            # Step 3. Perform tbe actual grid search!
            (
                best_alpha_last,
                best_alpha_exact,
                best_metric_last,
                best_metric_exact,
                improved_last,
                improved_exact,
            ) = self.perform_calibration(
                alpha_ranges=self.alpha_range,
                best_alpha_last=1.0,
                best_metric_last=ref_metric_value_last,
                best_alpha_exact=1.0,
                best_metric_exact=ref_metric_value_exact,
                improved_last=False,
                improved_exact=False,
                master_metrics=master_metrics,
                ref_metric_value_last=ref_metric_value_last,
                ref_metric_value_exact=ref_metric_value_exact,
                ref_y_correct=ref_y_correct,
                ref_y_correct_exact=ref_y_correct_exact,
                alpha_results_table=alpha_results_table,
                search_type="Base",
            )

        # FIXME
        # Step 4. Perform refined search!
        if (
            self.refine_best_alpha
            and self.best_alpha_last != 1.0
            or self.best_alpha_exact != 1.0
        ):

            refined_candidates = []
            for best_alpha in [best_alpha_last, best_alpha_exact]:
                low, high = self.find_neighbour_midpoints(
                    alpha=best_alpha, alpha_range=self.alpha_range
                )
                if low is not None:
                    refined_candidates.append(low)
                if high is not None:
                    refined_candidates.append(high)
            refined_candidates = sorted(np.unique(refined_candidates))
            print(
                f"[CALIBRATION] Refining alpha search with midpoints to closest neighbours... {refined_candidates}"
            )
            (
                best_alpha_last,
                best_alpha_exact,
                best_metric_last,
                best_metric_exact,
                improved_last,
                improved_exact,
            ) = self.perform_calibration(
                alpha_ranges=refined_candidates,
                best_alpha_last=best_alpha_last,
                best_metric_last=best_metric_last,
                best_alpha_exact=best_alpha_exact,
                best_metric_exact=best_metric_exact,
                improved_last=improved_last,
                improved_exact=improved_exact,
                master_metrics=master_metrics,
                ref_metric_value_last=ref_metric_value_last,
                ref_metric_value_exact=ref_metric_value_exact,
                ref_y_correct=ref_y_correct,
                ref_y_correct_exact=ref_y_correct_exact,
                alpha_results_table=alpha_results_table,
                search_type="Refined",
            )

        # Step 5. Apply decision rule!
        self.set_alpha_attrs(
            improved_last,
            ref_metric_value_last,
            best_alpha_last,
            best_metric_last,
            alpha_calibration_token_pos_target="last",
        )
        self.set_alpha_attrs(
            improved_exact,
            ref_metric_value_exact,
            best_alpha_exact,
            best_metric_exact,
            alpha_calibration_token_pos_target="exact",
        )

        print(
            f"[CALIBRATION] Best Last Alpha: {best_alpha_last} | {self.objective_key}: {best_metric_last:.4f}"
        )
        print(
            f"[CALIBRATION] Best Exact Alpha: {best_alpha_exact} | {self.objective_key}: {best_metric_exact:.4f}."
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

        self.disable_tdqm = False
        print("[INFO] Alpha Grid Search Completed.\n")

        return None

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

    def find_neighbour_midpoints(
        self, alpha: float, alpha_range: List[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find one midpoint with the closest lower and one with the closest upper neighbour of alpha."""
        sorted_alphas = sorted(alpha_range)
        lower = max((a for a in sorted_alphas if a < alpha), default=None)
        upper = min((a for a in sorted_alphas if a > alpha), default=None)

        lower_midpoint = (alpha + lower) / 2 if lower is not None else None
        upper_midpoint = (alpha + upper) / 2 if upper is not None else None

        return lower_midpoint, upper_midpoint

    def perform_calibration(
        self,
        alpha_ranges: list,
        best_alpha_last: float,
        best_metric_last: float,
        best_alpha_exact: float,
        best_metric_exact: float,
        improved_last: bool,
        improved_exact: bool,
        master_metrics: dict,
        ref_metric_value_last: float,
        ref_metric_value_exact: float,
        ref_y_correct: float,
        ref_y_correct_exact: float,
        alpha_results_table,
        search_type: str,
        early_stopping: bool = False,
        patience: int = 5,
        min_alpha_steps: int = 3,
    ):
        """Perform the actual grid search!"""
        steps_last = 0
        steps_exact = 0

        for alpha_step, alpha in tqdm(
            enumerate(alpha_ranges),
            desc=f"{search_type} Grid Search Alpha...",
            leave=True,
            disable=False,
        ):

            self.disable_tdqm = False

            # Evaluate with current alpha!
            print(
                f"[CALIBRATION] Evaluating with current alpha_value: {alpha:.3f} ... "
            )
            print(
                f"[INFO] Evaluating with objective: {self.objective_key} and {self.objective_key_exact}"
            )

            metrics = self.evaluate(
                prompts=self.ref_prompts,
                labels=self.ref_labels,
                alpha_value=alpha,
                disable_tdqm=True,
                prefix="inner_evaluation/",
            )

            # Add metrics.
            master_metrics[alpha] = metrics
            current_metric_last = metrics.get(
                f"{self.prefix}{self.objective_key}", None
            )
            current_metric_exact = metrics.get(
                f"{self.prefix}{self.objective_key_exact}", None
            )

            if current_metric_last is None or current_metric_exact is None:
                raise ValueError(f"Evaluation metrics are missing expected keys.")

            # Compute deltas.
            delta = (
                current_metric_last - ref_metric_value_last
                if (
                    ("Accuracy" in self.objective_key)
                    or ("F1 Score" in self.objective_key)
                    or ("Recall" in self.objective_key)
                    or ("Precision" in self.objective_key)
                )
                else ref_metric_value_last - current_metric_last
            )
            delta_exact = (
                current_metric_exact - ref_metric_value_exact
                if (
                    ("Accuracy Exact" in self.objective_key)
                    or ("F1 Score Exact" in self.objective_key)
                    or ("Recall Exact" in self.objective_key)
                    or ("Precision Exact" in self.objective_key)
                )
                else ref_metric_value_exact - current_metric_exact
            )

            # Update Best Metric!
            if (
                ("Accuracy" in self.objective_key)
                or ("F1 Score" in self.objective_key)
                or ("Recall" in self.objective_key)
                or ("Precision" in self.objective_key)
            ) and current_metric_last > best_metric_last:
                best_metric_last = current_metric_last
                best_alpha_last = alpha
                improved_last = True
                steps_last = 0
            elif (
                "Error" in self.objective_key and current_metric_last < best_metric_last
            ):
                best_metric_last = current_metric_last
                best_alpha_last = alpha
                improved_last = True
                steps_last = 0
            else:
                if alpha_step >= min_alpha_steps - 1:
                    steps_exact += 1

            # Update Best Metric
            if (
                ("Accuracy Exact" in self.objective_key)
                or ("F1 Score Exact" in self.objective_key)
                or ("Recall Exact" in self.objective_key)
                or ("Precision Exact" in self.objective_key)
            ) and current_metric_exact > best_metric_exact:
                best_metric_exact = current_metric_exact
                best_alpha_exact = alpha
                improved_exact = True
                steps_exact = 0
            elif (
                "Error" in self.objective_key_exact
                and current_metric_exact < best_metric_exact
            ):
                # and current_metric_exact > np.sqrt(np.log(2/0.05)/(2*self.nr_samples)):
                best_metric_exact = current_metric_exact
                best_alpha_exact = alpha
                improved_exact = True
                steps_exact = 0
            else:
                if alpha_step >= min_alpha_steps - 1:
                    steps_exact += 1

            print(
                f"[CALIBRATION] Alpha: {alpha:.3f} | Current Metric Last: {current_metric_last:.4f} | Delta Last: {delta:.4f}"
            )
            print(
                f"[CALIBRATION] Alpha: {alpha:.3f} | Current Metric Exact: {current_metric_exact:.4f} | Delta Exact: {delta_exact:.4f}"
            )

            # Add to wandb table.
            alpha_results_table.add_data(
                alpha,
                search_type,
                ref_metric_value_last,
                current_metric_last,
                delta,
                np.sum(metrics[f"{self.prefix}Correct Predictions"])
                - np.sum(ref_y_correct),
                (
                    (
                        np.sum(metrics[f"{self.prefix}Correct Predictions"])
                        / np.sum(ref_y_correct)
                    )
                    if np.sum(ref_y_correct) != 0
                    else 0.0
                ),
                ref_metric_value_exact,
                current_metric_exact,
                delta_exact,
                np.sum(metrics[f"{self.prefix}Correct Predictions Exact"])
                - np.sum(ref_y_correct_exact),
                (
                    (
                        np.sum(metrics[f"{self.prefix}Correct Predictions Exact"])
                        / np.sum(ref_y_correct_exact)
                    )
                    if np.sum(ref_y_correct_exact) != 0
                    else 0.0
                ),
            )
            if np.isclose(delta, 0, rtol=1e-3) and steps_last >= min_alpha_steps - 1:
                steps_last += 1
            if (
                np.isclose(delta_exact, 0, rtol=1e-3)
                and steps_exact >= min_alpha_steps - 1
            ):
                steps_exact += 1

            if early_stopping and steps_last >= patience and steps_exact >= patience:
                print(
                    f"[CALIBRATION] Early stopping! No change identified in Last and Exact for {patience} steps."
                )
                break

        print(
            f"[CALIBRATION] Best Last Alpha: {best_alpha_last:.3f} | Best Metric Last: {best_metric_last:.4f} | Improved Last: {improved_last}"
        )
        print(
            f"[CALIBRATION] Best Exact Alpha: {best_alpha_exact:.3f} | Best Metric Exact: {best_metric_exact:.4f} | Improved Exact: {improved_exact}"
        )

        return (
            best_alpha_last,
            best_alpha_exact,
            best_metric_last,
            best_metric_exact,
            improved_last,
            improved_exact,
        )

    def get_alpha_results(self) -> List[Tuple[float, float]]:
        """
        Retrieve the results from the alpha grid search.

        Returns:
            List of tuples (alpha, metric_value).
        """
        if hasattr(self, "alpha_search_results"):
            return self.alpha_search_results
        else:
            raise ValueError(
                "No alpha search results found. Run calibration_alpha() first."
            )
