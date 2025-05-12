import pickle
from datetime import datetime
import torch
import numpy as np
import argparse
from typing import List, Dict, Optional, Callable, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch.nn.functional as F
from utils import *
from cache.cache_utils import *
from tasks.task_handler import *


def main(
    hf_token: str,
    token_position: int,
    cache_dir: str,
    save_dir: str,
    model_name: str,
    dataset_names: List[str],
    nr_samples: int,
    batch_size: int,
    device: str,
    n_devices: int,
    run_acts: bool,
    run_saes: bool,
    flexible_match: bool,
    overwrite: bool,
    save: bool = True,
    verbose: bool = True,
) -> None:

    for dataset_name in dataset_names:

        task_config = TaskConfig(
            token=hf_token,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            nr_samples=nr_samples,
            model_name=model_name,
            device=device if device != "" else None,
            nr_devices=n_devices,
            flexible_match=flexible_match,
            batch_size=batch_size,
        )

        model_handler = ModelHandler(task_config)
        dataset_handler = DatasetHandler(task_config, tokenizer=model_handler.tokenizer)
        nr_layers = model_handler.nr_layers

        today = datetime.today().strftime("%Y%m%d")[2:]
        save_dir_curr = save_dir + f"{dataset_name}/"
        save_dir_curr += f"{model_name.split('/')[1]}/"
        os.makedirs(save_dir_curr, exist_ok=True)  # uuid.uuid1().hex[:3]uuid
        save_key = f"{nr_samples}_"

        '''
        for i, p in enumerate(dataset_handler.prompts):
            print(p)
            if i == 5:
                break
        '''
        
        # Step 1: Generate completions.
        completions = generate_completions(
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            tokenizer_kwargs=model_handler.tokenizer_kwargs,
            prompts=dataset_handler.prompts,
            dataset_info=dataset_handler.dataset_info,
            batch_size=batch_size,
            device=model_handler.model.device,
            flexible_match=task_config.flexible_match,
            save_dir=save_dir_curr,
            save_key=save_key,
            overwrite=overwrite,
        )

        # Step 2: Compute logits.
        logits, softmaxs = compute_logits(
            model=model_handler.model,
            completions=completions["completions"],
            # attention_mask=completions["attention_masks"],
            mode="all",
            position=None,
            flexible_match=task_config.flexible_match,
            dataset_info=dataset_handler.dataset_info,
            save_dir=save_dir_curr,
            save_key=save_key,
            overwrite=overwrite,
        )
        clean_gpus()

        # Step 3: Compute and save targets.
        targets = compute_targets(
            y_softmax_all=softmaxs,
            y_true=dataset_handler.y_true,
            dataset_info=dataset_handler.dataset_info,
            prompt_sequence_lengths=completions["prompt_sequence_lengths"],
            match_indices=completions["match_indices"],
            save_dir=save_dir_curr,
            save_key=save_key,
            overwrite=overwrite,
        )
        clean_gpus()

        print(f'SM Error last {np.mean(1-np.array(targets["y_softmax"]))}')
        print(f'SM Error exact {1-np.mean(np.array(targets["y_softmax_exact"]))}')
        print(f'Correct last {np.mean(targets["y_correct"])}')
        print(f'Correct exact {np.mean(targets["y_correct_exact"])}')

        if run_acts:

            # Step 4: Generate activations.
            activations = collect_activations(
                model=model_handler.model,
                completions=completions["completions"],
                batch_size=batch_size,
                mode="all",
                nr_layers=nr_layers,
                save_dir=save_dir_curr,
                save_key=save_key,
                overwrite=overwrite,
            )

        if run_saes:

            # Ugly solution, but only import SAE lens if necessary.
            from cache.cache_sae_utils import collect_saes

            # Step 5: Generate SAE activations.
            sae_activations = collect_saes(
                model_name=task_config.model_name,
                tokenizer=model_handler.tokenizer,
                tokenizer_kwargs=model_handler.tokenizer_kwargs,
                prompts=completions["completions_str"],
                cache_dir=cache_dir,
                batch_size=batch_size,
                n_devices=6,
                nr_layers=nr_layers,
                save_dir=save_dir_curr,
                save_key=save_key,
                overwrite=overwrite,
                hf_token=hf_token,
            )

    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Distributed processing for Gemma model task."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="None",
        help="Huggingface token.",
    )
    parser.add_argument(
        "--token_position",
        type=int,
        default=None,
        help="Position of the token to analyse. If None, selects all tokens.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../hf-cache/",
        help="File path for getting the data.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../runs/",
        help="File path for getting the data.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b-it",
        help="Name of the model to load.",   )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=[
            # "sentiment_analysis",
            # "mmlu_high_school",
            # "mmlu_professional",
             "sms_spam",
            # "yes_no_question",
        ],
        help="Name(s) of the task(s) to load. Provide a single name or a space-separated list.",
    )
    parser.add_argument(
        "--nr_samples", type=int, default=3000, help="Number of samples to process."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch_size to process."
    )
    parser.add_argument("--n_devices", type=int, default=6, help="Number of devices.")
    parser.add_argument(
        "--flexible_match",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If run in with flexible matching.",
    )
    parser.add_argument(
        "--overwrite",
        default=True, # FIXME
        action=argparse.BooleanOptionalAction,
        help="If overwrite existing results.",
    )
    parser.add_argument(
        "--run_acts",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If run_acts existing results.",
    )
    parser.add_argument(
        "--run_saes",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If run_saes existing results.",
    )
    parser.add_argument(
    "--device",
    type=str,
    default="",
    help="Device for single GPU use (default: cuda:0).",
)

    args = parser.parse_args()

    main(
        args.hf_token,
        args.token_position,
        args.cache_dir,
        args.save_dir,
        args.model_name,
        args.dataset_names,
        args.nr_samples,
        args.batch_size,
        args.device,
        args.n_devices,
        args.run_acts,
        args.run_saes,
        args.flexible_match,
        args.overwrite,
    )
