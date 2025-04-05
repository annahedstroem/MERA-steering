import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
import pickle

from cache.cache_utils import load_saved_data
from steering.constants import *
from utils import *


def train_probes(
    features: dict,
    models: dict,
    targets: dict,
    metrics: dict,
    dataset_name: str,
    llm_name: str,
    path: str,
    path_probes: str,
    nr_layers,
    error_type: str = "",
    seed: int = 42,
    nr_models: int = 5,
    max_attempts: int = 3,
    epsilon: float = 1e-10,
    save: bool = True,
    transform_error: bool = False,
    normalise_error: bool = False,
    token_pos: str = "",
) -> pd.DataFrame:
    """Train linear probes on LM activations using classification and regression targets,
    with support for multiple token positions, feature types (e.g. activations, SAEs),
    model types (Lasso, Logistic, etc.), and evaluation metrics. Applies transformations
    (e.g. log-odds) or normalisation to regression targets and evaluates feature sparsity
    via coefficient selection across layers with repeated initialisations and dummy baselines.
    """

    selected_layers = list(np.arange(nr_layers))
    probing_results = []
    model_objects = {}

    for feature_name, layer_data in tqdm(features.items(), desc="Processing Features"):
        for layer_idx, (layer_name, layer_features) in tqdm(
            enumerate(layer_data.items()), desc=f"{feature_name} Layers"
        ):
            if selected_layers and layer_name not in selected_layers:
                continue
            X = np.array(layer_features)
            for model_task, y_true in targets.items():

                # Transform to log-odds-ratio space (numerical stability step).
                if transform_error and model_task == "regression":
                    y_true = np.clip(y_true, 1e-8, 1 - 1e-8)
                    y_true = np.log(y_true / (1 - y_true))

                elif normalise_error and model_task == "regression":
                    y_true /= y_true.max()

                if model_task == "classification":
                    y_true = [y for y in y_true]

                for model_name, model in models[model_task].items():

                    # Train several linear models and add it as one row to the df.
                    for m in range(nr_models):

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_true, test_size=0.3, random_state=seed + m
                        )

                        # Dummy baseline for comparison!
                        dummy_model = (
                            DummyClassifier(strategy="most_frequent")
                            if model_task == "classification"
                            else DummyRegressor(strategy="mean")
                        )
                        dummy_model.fit(X_train, y_train)
                        dummy_y_pred = dummy_model.predict(X_test)
                        dummy_metrics = {
                            f"Dummy-{metric}": metrics[model_task][metric](
                                y_test, dummy_y_pred
                            )
                            for metric in metrics[model_task]
                        }

                        no_coeffs = True
                        attempt = 0

                        while no_coeffs and attempt < max_attempts:
                            attempt += 1
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # Collect coefficients and count non-zero ones.
                            coeffs = []
                            used_indices = []
                            if hasattr(model, "coef_"):
                                coeffs = model.coef_.flatten().tolist()
                                used_indices = np.where(np.abs(model.coef_) > epsilon)[
                                    0
                                ].tolist()

                            no_coeffs = (
                                len(used_indices) == 0
                            )  # True if no features are non-zero

                            # Compute residuals!
                            residuals = (
                                abs(y_test - y_pred).tolist()
                                if model_task == "regression"
                                else []
                            )

                            # Compute metrics!
                            res = {
                                **{
                                    metric: metrics[model_task][metric](y_test, y_pred)
                                    for metric in metrics[model_task]
                                },
                                **dummy_metrics,
                            }

                            if not no_coeffs:

                                # Append results!
                                probing_results.append(
                                    {
                                        "Dataset": dataset_name,
                                        "LLM_model": llm_name,
                                        "Task": model_task,
                                        "Model": model_name,
                                        "Inputs": feature_name,
                                        "Error-Type": error_type,
                                        "Layer": layer_idx,
                                        "Residuals": residuals,
                                        "Coefficients": coeffs,
                                        "Nonzero-Features": used_indices,
                                        "No-Coefficients": no_coeffs,
                                        "Attempt": attempt,
                                        "Model-Index": m + 1,
                                        "Token-Pos": token_pos,
                                        "y_pred": y_pred.tolist(),
                                        "y_test": y_test,
                                        **res,
                                    }
                                )

                        # Save model object!
                        model_key = f"{dataset_name}_{model_task}_{model_name}_{feature_name}_{layer_name}_model{m+1}".lower()
                        model_objects[model_key] = model

    df_probes = pd.DataFrame(probing_results)
    if save:
        df_probes.to_pickle(path.replace(".pkl", f"_{dataset_name}.pkl"))
        with open(path_probes.replace(".pkl", f"_{dataset_name}.pkl"), "wb") as f:
            try:
                pickle.dump(model_objects, f)
            except Exception as e:
                print(f"Could not fully load file {f} — {e}")

    return df_probes, model_objects


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train probes to use for steering.")
    parser.add_argument("--nr_layers", type=int, default=36, help="Number of layers.")
    parser.add_argument("--seed", type=int, default=52, help="Experiment seed.")
    parser.add_argument(
        "--save_name", type=str, default="", help="Extra name for saving probe."
    )
    parser.add_argument(
        "--save_cache_key", type=str, default="3000", help="Save key for the cache."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../runs",
        help="Save directory to retrieve the cache.",
    )
    parser.add_argument(
        "--token_pos", nargs="+", default=["", "_exact"], help="List of token_pos."
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=[
            # "sentiment_analysis",
            # "mmlu_high_school",
            "mmlu_professional",
            # "sms_spam",
            # "yes_no_question",
        ],
        help="Task names.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "google/gemma-2-2b-it",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-3B-Instruct",
            "google/gemma-2-2b", 
            
        ],
        help="Models to include (e.g., Qwen/Qwen2.5-3B-Instruct).",
    )
    parser.add_argument(
        "--process_saes",
        type=str,
        default="False",
        help="Enable or disable SAEs processing.",
    )
    parser.add_argument(
        "--transform_targets",
        type=str,
        default="True",
        help="Enable or disable transformation on targets.",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")

    (
        process_saes,
        transform_targets,
        nr_layers,
        dataset_names,
        save_name,
        seed,
        token_pos_all,
        save_cache_key,
        save_dir,
    ) = (
        args.process_saes.lower() == "true",
        args.transform_targets.lower() == "true",
        args.nr_layers,
        args.dataset_names,
        args.save_name,
        args.seed,
        args.token_pos,
        args.save_cache_key,
        args.save_dir,
    )

    dataset_names = filter_valid(SUPPORTED_TASKS, args.dataset_names)
    model_names = filter_valid(SUPPORTED_MODELS, args.model_names)
    print(f"[INFO] process_saes = {process_saes}.")
    print(f"[INFO] transform_targets = {transform_targets}.")

    metrics = {
        "regression": {
            "RMSE": lambda y_true, y_pred: mean_squared_error(
                y_true, y_pred, squared=False
            ),
            "MSE": mean_squared_error,
        },
        "classification": {
            "AUCROC": roc_auc_score,
            "Accuracy": accuracy_score,
            "Accuracy (Balanced)": balanced_accuracy_score,
        },
    }

    def initialise_regression_models(seed: int, alphas) -> dict:
        """Initialise regression models with various hyperparameters."""
        models = {
            f"L-{alpha}": Lasso(
                alpha=alpha, fit_intercept=False, max_iter=2000, random_state=seed
            )
            for alpha in alphas
        }  # positive=True,
        models["L-0"] = LinearRegression(fit_intercept=False, n_jobs=5)
        return models

    def initalise_classification_models(seed: int) -> dict:
        models = {
            "LogReg-l1": LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=2000,
                fit_intercept=False,
                random_state=seed,
            )
        }
        return models

    alphas = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
    models = {
        "classification": initalise_classification_models(seed),
        "regression": initialise_regression_models(seed, alphas),
    }
    error_type = "sm"
    save_name = f"_{save_name}" if save_name != "" else save_name

    for model_name in model_names:
        print(f"Processing {model_name}...")

        for dataset_name in dataset_names:
            print(f"Processing {dataset_name}...")

            if "gemma" in model_name:
                nr_layers = 26
            elif "Qwen" in model_name:
                nr_layers = 36
            elif "Llama" in model_name:
                nr_layers = 16
            print(f"[DEBUG] 'nr_layers' set to {nr_layers}.")

            list_probes = []
            model_objects = {}

            # How to save the probes.
            path_probe = f'{save_dir}/probes/sub/df_probes_{model_name.split("/")[1]}{save_name}.pkl'
            path_probe_models = f'{save_dir}/probes/sub/models_{model_name.split("/")[1]}{save_name}.pkl'
            path_df = f"{save_dir}/{dataset_name}/{model_name.split('/')[1]}/df_probes{save_name}.pkl"

            # Get the postprocessed data.
            k = "_with_saes" if process_saes else ""
            file_path = f"{save_dir}/{dataset_name}/{model_name.split('/')[1]}/{save_cache_key}_acts{k}.pkl"

            # Load activations.
            with open(file_path, "rb") as f:
                try:
                    acts = pickle.load(f)
                except Exception as e:
                    print(f"Could not fully load file {f} — {e}")

            # Load targets.
            y_targets = load_saved_data(
                save_dir=f"{save_dir}/{dataset_name}/{model_name.split('/')[1]}/",
                save_key=save_cache_key,
                data_type="targets",
            )

            for token_pos in token_pos_all:

                # Prepare the features.
                acts_cache = acts[f"activations_cache{token_pos}"]
                if process_saes:
                    sae_enc_cache = acts[f"sae_enc_cache{token_pos}"]
                features = {
                    "activations": acts_cache,
                }
                if process_saes:
                    features["encodings"] = sae_enc_cache

                # Load task-specific targets.
                y_correct = [
                    int(pred == true)
                    for pred, true in zip(
                        y_targets[f"y_pred{token_pos}"], y_targets["y_true"]
                    )
                ]
                if error_type == "ce":
                    y_error = y_targets[f"y_error{token_pos}"]
                else:
                    y_error = 1 - np.array(y_targets[f"y_softmax{token_pos}"])

                # Prepare the targets.
                targets = {
                    "classification": y_correct,
                    "regression": y_error,
                }

                # Train probes!
                df_probes, model_objects = train_probes(
                    features,
                    models,
                    targets,
                    metrics,
                    nr_models=5,
                    llm_name=model_name,
                    dataset_name=dataset_name.replace(".pkl", ""),
                    error_type=error_type.replace("_", ""),
                    nr_layers=nr_layers,
                    path=path_probe,
                    path_probes=path_probe_models,
                    token_pos=token_pos + "last" if token_pos == "" else "exact",
                    normalise_error=False,
                    transform_error=transform_targets,
                )
                list_probes.append(df_probes)
                model_objects.update(model_objects)

            # Save per model for both exact and last (!)
            df_probes_full = pd.concat(list_probes)
            df_probes_full.to_pickle(path_df)
            df_probes_full.to_pickle(path_probe.replace("sub/", ""))

            # Save the trained models.
            with open(path_probe_models.replace("sub/", ""), "wb") as f:
                try:
                    pickle.dump(model_objects, f)
                except Exception as e:
                    print(f"Could not fully load file {f} — {e}")
