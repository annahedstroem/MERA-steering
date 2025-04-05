import argparse
import pickle
from datetime import datetime
from tqdm import tqdm
import numpy as np

from tasks.task_handler import *
from steering.constants import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Postprocess cache to use for probe training."
    )
    parser.add_argument("--nr_layers", type=int, default=26, help="Number of layers.")
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
        "--task_names",
        nargs="+",
        default=[
            "sentiment_analysis",
             "mmlu_high_school",
            # "mmlu_professional",
             "sms_spam",
             "yes_no_question",
        ],
        help="Task names.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
             "google/gemma-2-2b-it",
             "google/gemma-2-2b",
             "meta-llama/Llama-3.2-1B-Instruct",
             "meta-llama/Llama-3.2-1B",
             "Qwen/Qwen2.5-3B-Instruct",
             "Qwen/Qwen2.5-3B",
        ],
        help="Models to include (e.g., Qwen/Qwen2.5-3B-Instruct).",
    )
    parser.add_argument(
        "--process_saes",
        type=str,
        default="False",
        help="Enable or disable SAEs processing.",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")

    process_saes, nr_layers, task_names, save_cache_key, save_dir = (
        args.process_saes.lower() == "true",
        args.nr_layers,
        args.task_names,
        args.save_cache_key,
        args.save_dir,
    )
    model_names = filter_valid(SUPPORTED_MODELS, args.model_names)

    results = {}

    for model_name in model_names:
        print(f"Processing {model_name}...")

        for task_name in task_names:
            print(f"Processing {task_name}...")

            if process_saes:
                sae_enc_layers_data = {}
                sae_dec_layers_data = {}
                sae_enc_layers_data_exact = {}
                sae_dec_layers_data_exact = {}

            act_layer_data = {}
            act_layer_data_exact = {}
            results[task_name] = {}

            file_path_cache = f"{save_dir}/{task_name}/{model_name.split('/')[1]}/"

            if "gemma" in model_name:
                nr_layers = 26
            elif "Qwen" in model_name:
                nr_layers = 36
            elif "Llama" in model_name:
                nr_layers = 16
            print(f"[DEBUG] 'nr_layers' set to {nr_layers}.")

            for layer in tqdm(
                range(nr_layers), desc=f"Processing Layers for {task_name}"
            ):

                if layer == 0:

                    with open(
                        f"{file_path_cache}{save_cache_key}_completions.pkl", "rb"
                    ) as f:
                        completions = pickle.load(
                            f
                        )  # torch.load(f, map_location="cuda:0")

                    with open(
                        f"{file_path_cache}{save_cache_key}_targets.pkl", "rb"
                    ) as f:
                        targets = pickle.load(f)  # torch.load(f, map_location="cuda:0")

                    last_matches = completions["prompt_sequence_lengths"]
                    exact_matches = completions["match_indices"]

                    y_error_sm = 1 - np.array(targets["y_softmax"])
                    y_correct = targets["y_correct"]
                    y_error_ce = targets["y_error"]
                    y_error_sm_exact = 1 - np.array(targets["y_softmax_exact"])
                    y_correct_exact = targets["y_correct_exact"]
                    y_error_ce_exact = targets["y_error_exact"]

                # Read inputs.
                if process_saes:
                    with open(
                        f"{file_path_cache}saes/{save_cache_key.replace('parse', '')}_sae_{layer}.pkl",
                        "rb",
                    ) as f:
                        sae_layer = pickle.load(
                            f
                        )  # sae_layer = torch.load(f, map_location="cuda:0")#
                with open(
                    f"{file_path_cache}activations/{save_cache_key.replace('parse', '')}_activations_{layer}.pkl",
                    "rb",
                ) as f:
                    act_layer = pickle.load(
                        f
                    )  # act_layer = torch.load(f, map_location="cuda:0") #

                # Get last.
                act_layer_data[layer] = np.vstack(
                    [
                        sample[last_match]
                        for sample, last_match in zip(act_layer, last_matches)
                    ]
                )
                if process_saes:
                    sae_enc_layers_data[layer] = np.vstack(
                        [
                            sample[exact_match - 1]
                            for sample, exact_match in zip(
                                sae_layer["sae_enc_cache"], last_matches
                            )
                        ]
                    )
                    sae_dec_layers_data[layer] = np.vstack(
                        [
                            sample[exact_match - 1]
                            for sample, exact_match in zip(
                                sae_layer["sae_dec_cache"], last_matches
                            )
                        ]
                    )

                # Get exact.
                act_layer_data_exact[layer] = np.vstack(
                    [
                        sample[exact_match]
                        for sample, exact_match in zip(act_layer, exact_matches)
                    ]
                )
                if process_saes:
                    sae_enc_layers_data_exact[layer] = np.vstack(
                        [
                            sample[exact_match - 1]
                            for sample, exact_match in zip(
                                sae_layer["sae_enc_cache"], exact_matches
                            )
                        ]
                    )  # if match == -1
                    sae_dec_layers_data_exact[layer] = np.vstack(
                        [
                            sample[exact_match - 1]
                            for sample, exact_match in zip(
                                sae_layer["sae_dec_cache"], exact_matches
                            )
                        ]
                    )

                if process_saes:
                    del sae_layer
                del act_layer

            results[task_name]["activations_cache"] = act_layer_data
            if process_saes:
                results[task_name]["sae_enc_cache"] = sae_enc_layers_data
                results[task_name]["sae_dec_cache"] = sae_dec_layers_data
            results[task_name]["y_correct"] = y_correct
            results[task_name]["y_error_sm"] = y_error_sm
            results[task_name]["y_error_ce"] = y_error_ce

            results[task_name]["activations_cache_exact"] = act_layer_data_exact
            if process_saes:
                results[task_name]["sae_enc_cache_exact"] = sae_enc_layers_data_exact
                results[task_name]["sae_dec_cache_exact"] = sae_dec_layers_data_exact
            results[task_name]["y_correct_exact"] = y_correct_exact
            results[task_name]["y_error_sm_exact"] = y_error_sm_exact
            results[task_name]["y_error_ce_exact"] = y_error_ce_exact

            k = "_with_saes" if process_saes else ""
            file_path_save = f"{save_dir}/{task_name}/{model_name.split('/')[1]}/{save_cache_key}_acts{k}.pkl"

            with open(file_path_save, "wb") as f:
                pickle.dump(results[task_name], f)
