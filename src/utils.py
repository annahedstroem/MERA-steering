import torch
import sys
import psutil
import gc

from difflib import get_close_matches

def filter_valid(options, selection):
    """Filter and validate selection against available options."""
    if not selection:
        return options  # Use all if nothing is specified.

    valid_selection = [x for x in selection if x in options]
    invalid_selection = [x for x in selection if x not in options]

    if invalid_selection:
        print(
            f"\nWARNING: The following choices are not valid: {', '.join(invalid_selection)}"
        )

        suggestions = {
            invalid: get_close_matches(invalid, options, n=1, cutoff=0.6)
            for invalid in invalid_selection
        }
        for method, suggestion in suggestions.items():
            if suggestion:
                print(f"Did you mean `{suggestion[0]}` instead of `{method}`?")

        print(f"\nAvailable choices:{', '.join(options)}.")

        if not valid_selection:
            print("\nNo valid choices selected. Exiting.\n")
            sys.exit(1)

    return valid_selection


def list_vars_by_size() -> list:
    return sorted(
        ((name, sys.getsizeof(value)) for name, value in globals().items()),
        key=lambda x: x[1],
        reverse=True,
    )


def memory_info_gb() -> tuple:
    memory_info = psutil.virtual_memory()
    available_memory_gb = memory_info.available / (1024**3)
    total_memory_gb = memory_info.total / (1024**3)
    return print(
        f"Available GB memory: {available_memory_gb:.2f} by total: {total_memory_gb:.2f}"
    )


def clean_gpus() -> None:
    gc.collect()
    torch.cuda.empty_cache()
