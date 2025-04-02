import os
import json
import pickle
import torch
from contextlib import suppress
from torch.cuda.amp import autocast
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple, Union, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.functional as F
from utils import *
from tasks.task_handler import *


def find_first_exact_match(
    completions: torch.Tensor,
    token_start_position: int,
    flexible_match: bool,
    dataset_info: dict,
    fallback: Optional[int] = None,
) -> Tuple[List[int], List[int], List[bool]]:
    """Find the position of the first match token for each sequence based on class labels.

    Args:
        completions: Tensor of token IDs from model completions.
        token_start_position: Starting position in the token sequence.
        flexible_match: Whether to match using all token IDs or exact ones.
        dataset_info: Dataset metadata containing 'VALID_GROUND_TRUTH_TOKEN_IDS'.
        fallback: Optional fallback index if no exact match is found.

    Returns:
        match_tokens: List of matched token IDs (or fallback token if no match).
        match_indices: List of positions of the first match (or fallback index).
        match_flags: List of booleans indicating whether an exact match was found.
    """
    if flexible_match:
        all_match_tokens = list(
            token_id
            for label, token_ids in dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"].items()
            for token_id in list(token_ids.values())
        )
    else:
        all_match_tokens = [
            token_ids[label]
            for label, token_ids in dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"].items()
        ]

    completions_token_ids = completions.cpu().numpy()
    match_indices = [-1] * len(completions)
    match_tokens = [-1] * len(completions)
    match_flags = [False] * len(completions)

    for idx, token_ids in enumerate(completions_token_ids):
        for pos in range(token_start_position, len(token_ids)):
            if token_ids[pos] in all_match_tokens:
                match_indices[idx] = pos
                match_tokens[idx] = token_ids[pos]
                match_flags[idx] = True  # Exact match found
                break

        # Fallback to the specified index if no match was found
        if not match_flags[idx] and fallback is not None:
            match_indices[idx] = fallback
            match_tokens[idx] = token_ids[fallback]

    return match_tokens, match_indices, match_flags


def get_ground_truth_valid_token_ids(dataset_info: Dict, flexible_match: bool) -> Dict:
    """Retrieve token IDs for each class based on dataset information."""
    VALID_GROUND_TRUTH_TOKEN_IDS = dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"]
    if flexible_match:
        return {
            cls: [token_id for token_id in class_ids.values()]
            for cls, class_ids in VALID_GROUND_TRUTH_TOKEN_IDS.items()
        }
    else:
        return {
            cls: (
                [VALID_GROUND_TRUTH_TOKEN_IDS[cls][cls]]
                if isinstance(VALID_GROUND_TRUTH_TOKEN_IDS[cls], dict)
                else [VALID_GROUND_TRUTH_TOKEN_IDS[cls]]
            )
            for cls in VALID_GROUND_TRUTH_TOKEN_IDS.keys()
        }


def get_logits(
    model: AutoModelForCausalLM,
    inputs: torch.Tensor,
    mode: str,
    position: Optional[int],
    grad: bool = False,
    use_cache: bool = False,
) -> torch.Tensor:
    """Compute logits based on the specified mode (last token, max_pool, position, or all)."""
    grad_context = torch.enable_grad() if grad else torch.no_grad()
    with grad_context:  # , autocast(dtype=torch.float16):
        outputs = model(inputs, use_cache=use_cache, return_dict=True)
        logits = outputs.logits
    clean_gpus()
    if mode == "last_token":
        return logits[:, -1, :].unsqueeze(1)  # Shape: [batch_size, 1, vocab_size]
    elif mode == "max_pool":
        return torch.max(
            logits, dim=1, keepdim=True
        ).values  # Shape: [batch_size, 1, vocab_size]
    elif mode == "position":
        assert position is not None, "Position must be specified for 'mode=position'."
        return logits[:, position, :].unsqueeze(1)  # Shape: [batch_size, 1, vocab_size]
    elif mode == "all":
        return logits  # Shape: [batch_size, seq_len, vocab_size]
    else:
        raise ValueError(
            "Invalid mode specified. Choose from 'last_token', 'max_pool', 'position', or 'all'."
        )


def aggregate_class_logits(
    logits: torch.Tensor, token_ids_per_class: Dict, agg_func: Callable
) -> torch.Tensor:
    """Aggregate logits for each class by applying the aggregation function to the token positions."""
    class_logits = []
    for token_ids in token_ids_per_class.values():
        class_logits_per_class = logits[:, :, token_ids]
        aggregated_logits, _ = agg_func(
            class_logits_per_class, dim=-1
        )  # FIXME. for gradient flow .cpu().numpy()
        class_logits.append(
            aggregated_logits
        )  # FIXME.  torch.tensor( ... .to(logits.device)
    class_logits = torch.stack(
        class_logits, dim=-1
    )  # Shape: [batch_size, seq_len (or 1), num_classes
    return class_logits


def compute_softmax_per_position(class_logits: torch.Tensor) -> torch.Tensor:
    """Compute softmax scores independently at each token position."""
    return torch.softmax(
        class_logits, dim=-1
    )  # class_logits = class_logits.to(dtype=torch.float32)


def calculate_cross_entropy_error(
    softmax_scores: List[np.ndarray],
    y_true: List[int],
    grad: bool,
) -> List[Union[np.ndarray, torch.Tensor]]:
    """Calculate cross-entropy error for each position in the sequence."""
    errors = []

    for softmax, true_label in zip(softmax_scores, y_true):
        seq_len = softmax.shape[0]
        num_classes = softmax.shape[1]

        if not (0 <= true_label < num_classes):
            raise ValueError(
                f"True label {true_label} is out of range for {num_classes} classes."
            )

        if not grad:
            one_hot_encoded = np.eye(num_classes)[true_label]
            log_softmax_scores = np.log(softmax + 1e-9)  # Avoid log(0) errors.
            cross_entropy_error = -np.sum(one_hot_encoded * log_softmax_scores, axis=-1)
        else:
            log_softmax = torch.log(softmax + 1e-9)
            true_label_tensor = torch.tensor(
                [true_label], dtype=torch.long, device=softmax.device
            )
            one_hot = F.one_hot(
                true_label_tensor, num_classes=softmax.shape[-1]
            ).float()
            cross_entropy_error = -torch.sum(one_hot * log_softmax, dim=-1)

        errors.append(cross_entropy_error if grad else cross_entropy_error)

    return errors


def get_activation_hook(
    layer_name: str, mode: str = "all", position: Optional[int] = None
):
    """
    Return a hook function to capture activations based on the specified mode.
    Modes:
        - "last_token": Capture activations at the last token position.
        - "max_pool": Capture the maximum activation across all tokens.
        - "position": Capture activations at a selected token position.
        - "all": Capture all the activations across all token positions.
    """

    def hook(module, input, output):
        if mode == "last_token":
            activations = output[:, -1, :]  # Last token activations.
        elif mode == "max_pool":
            activations, _ = torch.max(
                output, dim=1
            )  # Max activation across the sequence.
        elif mode == "position":
            assert (
                position is not None
            ), "Arg 'exact' must be an integer to run get_activation_hook with 'mode=select_position'."
            activations = output[:, position, :]
        elif mode == "all":
            activations = output

        if layer_name not in activations_cache:
            activations_cache[layer_name] = []
        activations_cache[layer_name].extend(activations.detach().cpu().numpy())

    return hook


def register_hooks(
    model: AutoModelForCausalLM,
    mode: str = "all",
    position: Optional[int] = None,
    nr_layers: int = 26,
):
    """Register hooks to capture activations with specified mode."""
    global hooks
    hooks = []

    for i, layer in enumerate(model.model.layers):
        layer_name = i  # f'post_attention_layernorm_{i}'
        hook = layer.post_attention_layernorm.register_forward_hook(
            get_activation_hook(layer_name, mode, position)
        )
        hooks.append(hook)

    return hooks


def deregister_hooks():
    """Deregister all hooks that were registered"""
    global hooks
    for hook in hooks:
        hook.remove()
    hooks = []


def collect_activations(
    model: AutoModelForCausalLM,
    completions: torch.tensor,
    nr_layers: int = 26,
    batch_size: int = 1,
    mode: str = "last_token",
    position: Optional[int] = None,
    save_dir: str = "../runs/",
    save_key: str = "activations",
    save: bool = True,
    overwrite: bool = True,
    use_cache: bool = False,
    disable_tdqm: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Collect activations for all layers for the given inputs, with specified pooling type and maximum length.

    Arguments:
        model: The language model.
        tokenizer: The tokenizer for the model.
        inputs: List of input prompts.
        batch_size: The size of the batch for processing.
        mode: Type of pooling to use ("last_token", "max_pool", "position" or "all").
    """
    save_dir += "activations"
    os.makedirs(save_dir, exist_ok=True)

    global activations_cache

    activations_cache = {i: [] for i in range(len(model.model.layers))}

    if not overwrite:
        nr_keys = 0
        for layer in activations_cache:
            file_path = f"{save_dir}/{save_key}activations_{layer}.pkl"
            if os.path.exists(file_path):
                nr_keys += 1
                with open(file_path, "rb") as f:
                    activations_cache[layer] = pickle.load(f)
        if nr_keys == len(activations_cache):
            print("All keys found for activations_cache, load instead.")
            return {}

    # Register hooks with specified pooling type.
    register_hooks(model, mode=mode, position=position, nr_layers=nr_layers)

    # Loop over the inputs in batches and tokenize with padding to max_length
    for i in tqdm(
        range(0, len(completions), batch_size),
        desc="Collecting Activations",
        disable=disable_tdqm,
        leave=True,
    ):
        batch_completions = completions[i : i + batch_size]
        input_tensor = (
            torch.tensor(batch_completions, dtype=torch.long)
            .to(model.device)
            .squeeze(0)
        )
        with torch.no_grad():
            model(input_tensor, use_cache=use_cache, return_dict=True)

    print("Deregistering hooks...")
    deregister_hooks()
    print("Hooks successfully deregistered.")

    if save:
        # Save all activations as a single file
        for layer in activations_cache:
            file_path = f"{save_dir}/{save_key}activations_{layer}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(activations_cache[layer], f)

        print(f"All activations saved to {file_path}")

    return activations_cache


def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenizer_kwargs: Dict,
    prompts: List[str],
    dataset_info: Dict,
    batch_size: int,
    device: torch.device,
    flexible_match: bool,
    save_dir: str,
    save_key: str,
    save: bool = True,
    overwrite: bool = True,
    grad: bool = False,
    use_cache: bool = True,
    disable_tdqm: bool = False,
    model_generation_kwargs: Optional[dict] = {},
) -> Dict[str, List]:
    """Generates completions and saves the result."""

    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/{save_key}completions.pkl"
    file_path_str = f"{save_dir}/{save_key}completions_str.pkl"

    if not overwrite:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                f1 = pickle.load(f)
            with open(file_path_str, "rb") as f:
                f2 = pickle.load(f)
            return {**f1, **f2}

    completions = []
    answers = []
    completions_str = []
    answers_str = []
    attention_masks = []
    match_tokens = []
    match_indices = []
    match_flags = []
    prompt_sequence_lengths = []
    completion_sequence_lengths = []
    max_length = dataset_info["MAX_LENGTH"]
    max_new_tokens = dataset_info["MAX_NEW_TOKENS"]

    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc="Generating Completions",
        disable=disable_tdqm,
        leave=True,
    ):
        batch_prompts = prompts[i : i + batch_size]
        prompts_tokenized = tokenizer(
            batch_prompts,
            **tokenizer_kwargs,
            # return_tensors="pt",
            # padding="max_length",
            # truncation=True,
            # max_length=max_length,
        )
        input_ids = prompts_tokenized.input_ids.to(device)
        attention_mask = prompts_tokenized.attention_mask.to(device)
        prompt_sequence_length = input_ids.shape[1]

        # grad_context = torch.no_grad() if not grad else suppress()
        grad_context = torch.enable_grad() if grad else torch.no_grad()
        with grad_context:  # , autocast(dtype=torch.float16):
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )
            completions.append(output.sequences.cpu().numpy())
            completions_str.extend(
                tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            )
            attention_mask_completion = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (
                            attention_mask.size(0),
                            output.sequences.size(1) - attention_mask.size(1),
                        ),
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            attention_masks.append(attention_mask_completion.cpu().numpy())
            answer = output.sequences[:, input_ids.shape[1] :]
            answer_str = tokenizer.batch_decode(answer, skip_special_tokens=True)
            answers.append(answer.cpu().numpy())
            answers_str.append(answer_str)
            completion_sequence_length = output.sequences.shape[1]

            # Find matches with token-based function!
            match_token, match_index, match_flag = find_first_exact_match(
                # tokenizer=tokenizer,
                completions=output.sequences,
                token_start_position=prompt_sequence_length,
                flexible_match=flexible_match,
                dataset_info=dataset_info,
                fallback=-1,
            )
            match_tokens.append(match_token[0])
            match_indices.append(match_index[0])
            match_flags.append(match_flag[0])
            prompt_sequence_lengths.append(prompt_sequence_length)
            completion_sequence_lengths.append(completion_sequence_length)

    result_pkl = {
        "completions": completions,
        "answers": answers,
        "attention_masks": attention_masks,
        "match_tokens": match_tokens,
        "match_indices": match_indices,
        "match_flags": match_flags,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "prompt_sequence_lengths": prompt_sequence_lengths,
        "completion_sequence_lengths": completion_sequence_lengths,
    }

    result_str_pkl = {
        "completions_str": completions_str,
        "answers_str": answers_str,
    }

    if save:

        with open(file_path, "wb") as f:
            pickle.dump(result_pkl, f)

        print(f"All completions saved to {file_path}")

        # Save completions.
        with open(file_path_str, "wb") as f:
            pickle.dump(result_str_pkl, f)

        print(f"All completions_str saved to {file_path_str}")

    return {**result_str_pkl, **result_pkl}


def compute_logits(
    model: AutoModelForCausalLM,
    completions: List[np.ndarray],
    mode: str,
    position: Optional[int],
    flexible_match: bool,
    dataset_info: Dict,
    save_dir: str,
    save_key: str,
    save: bool = True,
    compute_class_logits: bool = True,
    overwrite: bool = True,
    grad: bool = False,
    use_cache: bool = False,
    disable_tdqm: bool = False,
) -> np.ndarray:
    """Computes logits for the completions and saves them."""
    prefix = "" if not compute_class_logits else "class_"
    file_path = f"{save_dir}{save_key}{prefix}logits.pkl"
    file_path_softmax = f"{save_dir}{save_key}{prefix}softmax.pkl"

    if not overwrite:
        if os.path.exists(file_path) and os.path.exists(file_path_softmax):
            with open(file_path, "rb") as f:
                f1 = pickle.load(f)
            with open(file_path_softmax, "rb") as f:
                f2 = pickle.load(f)
            return f1, f2

    logits = []
    softmaxs = []
    token_ids_per_class = get_ground_truth_valid_token_ids(dataset_info, flexible_match)

    for batch_completions in tqdm(
        completions,
        desc="Computing Logits and Softmax scores",
        disable=disable_tdqm,
        leave=True,
    ):

        batch_tensor = torch.tensor(batch_completions, dtype=torch.long).to(
            model.device
        )
        batch_logits = get_logits(
            model,
            batch_tensor,
            mode=mode,
            position=position,
            grad=grad,
            use_cache=use_cache,
        )

        if compute_class_logits:
            batch_logits = aggregate_class_logits(
                batch_logits, token_ids_per_class, agg_func=torch.max
            )

        batch_softmaxs = compute_softmax_per_position(batch_logits)

        if not grad:
            logits.append(batch_logits.cpu().numpy().squeeze())
            softmaxs.append(batch_softmaxs.cpu().numpy().squeeze())
        else:
            logits.append(batch_logits.squeeze())
            softmaxs.append(batch_softmaxs.squeeze())

    if save:
        with open(file_path, "wb") as f:
            pickle.dump(logits, f)

        print(f"All logits saved to {file_path}")

        with open(file_path_softmax, "wb") as f:
            pickle.dump(softmaxs, f)

        print(f"All softmaxs saved to {file_path_softmax}")

    return logits, softmaxs


def compute_targets(
    y_softmax_all: List[np.ndarray],
    y_true: List[int],
    prompt_sequence_lengths: List[int],
    dataset_info: dict,
    match_indices: List[int],
    save_dir: str,
    save_key: str,
    save: bool = True,
    overwrite: bool = True,
    grad: bool = False,
) -> Dict[str, List]:
    """Computes all targets (errors, predictions, correctness) and saves them."""

    class_index_to_label = dataset_info["CLASS_INDEX_TO_LABEL"]

    # print("class_index_to_label)")
    # print(class_index_to_label)
    # max_length = dataset_info["MAX_LENGTH"]
    file_path = f"{save_dir}{save_key}targets.pkl"

    if not overwrite:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)

    # Compute errors!
    y_error_all = calculate_cross_entropy_error(y_softmax_all, y_true, grad)
    y_error = [
        y_error[last_match]
        for y_error, last_match in zip(y_error_all, prompt_sequence_lengths)
    ]

    # FIXME. Assumes error to be 1 at max ...
    y_error_exact = [
        y_error[exact_match] if exact_match != -1 else 1
        for y_error, exact_match in zip(y_error_all, match_indices)
    ]

    # Compute predictions at last and exact.
    if grad:
        y_pred = [
            torch.argmax(y_softmax[last_match, :])
            for y_softmax, last_match in zip(y_softmax_all, prompt_sequence_lengths)
        ]
        y_pred_exact = [
            torch.argmax(y_softmax[exact_match, :]) if exact_match != -1 else -1
            for y_softmax, exact_match in zip(y_softmax_all, match_indices)
        ]
        y_pred_labels = [class_index_to_label[pos.item()] for pos in y_pred]
        y_pred_labels_exact = [
            class_index_to_label[pos.item()] if pos.item() != -1 else ""
            for pos in y_pred_exact
        ]
    else:
        y_pred = [
            np.argmax(y_softmax[last_match, :])
            for y_softmax, last_match in zip(y_softmax_all, prompt_sequence_lengths)
        ]
        y_pred_exact = [
            np.argmax(y_softmax[exact_match, :]) if exact_match != -1 else -1
            for y_softmax, exact_match in zip(y_softmax_all, match_indices)
        ]
        y_pred_labels = [class_index_to_label[pos] for pos in y_pred]
        y_pred_labels_exact = [
            class_index_to_label[pos] if pos != -1 else "" for pos in y_pred_exact
        ]

    # Compute correctness.
    y_correct = [pred == true for pred, true in zip(y_pred, y_true)]
    y_correct_exact = [pred == true for pred, true in zip(y_pred_exact, y_true)]
    # y_correct_all = [[pred == true for pred in preds] for preds, true in zip(y_pred_all, y_true) ]

    # Compute softmax as error.
    y_softmax = [
        scores[last_match, y]
        for scores, last_match, y in zip(y_softmax_all, prompt_sequence_lengths, y_true)
    ]
    y_softmax_exact = [
        scores[exact_match, y] if exact_match != -1 else 1
        for scores, exact_match, y in zip(y_softmax_all, match_indices, y_true)
    ]

    # Combine all y_* values into one structure.
    targets = {
        "y_true": y_true,
        "y_error_all": y_error_all,
        "y_error": y_error,
        "y_error_exact": y_error_exact,
        "y_pred": y_pred,
        "y_pred_exact": y_pred_exact,
        "y_pred_labels": y_pred_labels,
        "y_pred_labels_exact": y_pred_labels_exact,
        "y_correct": y_correct,
        "y_correct_exact": y_correct_exact,
        "y_softmax": y_softmax,
        "y_softmax_exact": y_softmax_exact,
    }

    if save:
        with open(file_path, "wb") as f:
            pickle.dump(targets, f)

        print(f"All targets saved to {file_path}")

    return targets


def load_saved_data(
    save_dir: str,
    save_key: str,
    data_type: str,
    layer: Optional[int] = None,
) -> Union[Dict[str, Any], np.ndarray]:
    """
    Loads specific saved data based on save_key, data_type, and optionally, a specific layer.

    Args:
        save_dir (str): Directory where the data is saved.
        save_key (str): Unique key for the saved data.
        data_type (str): Type of data to load. Options include:
            - "completions"
            - "completions_str"
            - "class_logits"
            - "class_softmax"
            - "targets"
            - "activations"
            - "sae"
        layer (Optional[int]): Specific layer to load. If None, loads the entire data.

    Returns:
        Union[Dict[str, Any], np.ndarray]: Loaded data, format depends on data_type and layer.
    """
    if data_type in {"activations", "sae_activations"}:
        save_dir += f"{data_type}"

    file_path = f"{save_dir}{save_key}_{data_type}"

    if data_type in {
        "class_logits",
        "class_softmax",
        "targets",
        "activations",
        "sae",
        "completions",
        "completions_str",
    }:
        # Load Pickle for these data types
        if data_type in {"activations", "sae"}:
            file_path += f"_{layer}"
        with open(f"{file_path}.pkl", "rb") as f:
            data = pickle.load(f)
    # elif data_type == "completions_str":
    #    # Load JSON for text data
    #    with open(f"{file_path}.json", "r") as f:
    #        data = json.load(f)
    else:
        raise ValueError(
            f"Unsupported data_type: {data_type}. Options are 'completions', 'logits', 'softmax_scores', 'targets', "
            f"'activations', 'sae_enc_cache', or 'sae_dec_cache'."
        )

    # Return full data if no layer is specified!
    return data
