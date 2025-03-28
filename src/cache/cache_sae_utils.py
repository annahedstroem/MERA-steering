from sae_lens import HookedSAETransformer, SAE



def collect_saes(
    model_name: str,
    tokenizer: AutoTokenizer,
    tokenizer_kwargs: Dict,
    prompts: list,
    cache_dir: str,
    n_devices: int,
    nr_layers: int,
    batch_size: int,
    save_dir: str,
    save_key: str,
    token_position: Optional[int] = None,
    run_reversed: bool = False,
    overwrite: bool = True,
    hf_token: Optional[str] = None,
    disable_tdqm: bool = False,
) -> None:
    """
    Collect and save SAE encodings and decodings for each layer of a model.

    Args:
        model_name (str): Name of the model to process.
        prompts (list): List of prompts to process.
        save_dir (str): Directory to save the results.
        cache_dir (str): Directory for cached models and data.
        n_devices (int): Number of devices to use for processing.
        nr_layers (int): Total number of layers to process.
        token_position (Optional[int]): Position of token to process. If None, process all tokens.
        run_reversed (bool): If True, process layers in reversed order.
    """
    save_dir += "saes"
    os.makedirs(save_dir, exist_ok=True)

    # Load SAE model
    if "gemma" in model_name:
        model_sae = HookedSAETransformer.from_pretrained(
            model_name.split("/")[1],
            tokenizer=tokenizer,
            device="cuda",
            n_devices=n_devices,
        )
    elif "llama" in model_name:
        model_sae = HookedSAETransformer.from_pretrained(
            model_name,
            tokenizer=tokenizer,
            device="cuda",
            n_devices=n_devices,
            from_pretrained_kwargs={"cache_dir": cache_dir, "token": hf_token},
        )
    clean_gpus()

    # Determine layer processing order.
    layer_order = range(nr_layers - 1, -1, -1) if run_reversed else range(nr_layers)

    for layer in tqdm(
        layer_order, desc="Processing Layers", disable=disable_tdqm, leave=True
    ):

        results = {"sae_enc_cache": [], "sae_dec_cache": []}

        file_path = f"{save_dir}/{save_key}sae_{layer}.pkl"

        if os.path.exists(file_path) and not overwrite:
            print(f"Layer {layer} already processed, skipping...")
            continue

        print(f"\nProcessing layer {layer}...")

        if "gemma" in model_name:
            release = "gemma-scope-2b-pt-res-canonical"
            sae_id = f"layer_{layer}/width_16k/canonical"
        elif "llama" in model_name:
            release = "llama_scope_lxr_8x"
            sae_id = f"l{layer}r_8x"
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device="cuda",
        )

        for i in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"Layer {layer} Prompts",
            disable=disable_tdqm,
            leave=True,
        ):

            batch_prompts = prompts[i : i + batch_size]

            clean_gpus()

            _, cache = model_sae.run_with_cache(
                batch_prompts, names_filter=[f"blocks.{layer}.hook_resid_post"]
            )
            cache.to(sae.device)

            sae_enc = sae.encode(cache[sae.cfg.hook_name])
            sae_dec = sae.decode(sae_enc)

            if token_position is not None:
                sae_enc_select = sae_enc[0, token_position]
                sae_dec_select = sae_dec[0, token_position]
            else:
                sae_enc_select = sae_enc[0, :]
                sae_dec_select = sae_dec[0, :]

            results["sae_enc_cache"].append(sae_enc_select.detach().cpu().numpy())
            results["sae_dec_cache"].append(sae_dec_select.detach().cpu().numpy())

        # Save intermediate results
        file_path = f"{save_dir}/{save_key}sae_{layer}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved intermediate results to {file_path}")

        del sae, results
        clean_gpus()

    print(f"All layers processed and results saved to {file_path}")

    return None