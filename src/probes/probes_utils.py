import pandas as pd
import numpy as np
from typing import Optional


def map_dataset(x: str) -> str:
    """Maps dataset string to standardized name."""
    x = x.lower().strip()
    if "mmlu" in x:
        if "high_school" in x:
            return "MMLU-HS"
        if "professional" in x:
            return "MMLU-Prof"
        return "MMLU"
    if "sms" in x:
        return "SMS SPAM"
    if "sentiment" in x:
        return "Sentiment"
    if "yes_no" in x:
        return "Yes_No"
    return "Unknown"


def postprocess_df_probes(
    df,
    filter_error_type: Optional[str] = "sm",
    filter_probe_token_pos: Optional[str] = "last",
    filter_inputs: Optional[str] = "activations",
) -> pd.DataFrame:
    """Process the probe performance result per LM model and apply filters at the end."""

    # Filter the df.
    if "Match-Type" in df.columns:
        df["Token-Pos"] = df["Match-Type"]
    for col, val, allowed in [
        ("Error-Type", filter_error_type, {"sm", "cm"}),
        ("Token-Pos", filter_probe_token_pos, {"last", "exact"}),
        ("Inputs", filter_inputs, {"activations", "encodings"}),
    ]:
        if val is not None:
            df[col] = df[col].str.lower()
            val = val.lower()
            assert val in allowed and val in df[col].unique()
            df = df[df[col] == val]

    # Dtypes and names.
    df["Dataset_name"] = df["Dataset"].apply(map_dataset)
    df["Layer"] = df["Layer"].astype(int)

    # Get into arrays.
    df["Residuals"] = df["Residuals"].apply(lambda x: np.array(x))
    df["Coefficients"] = df["Coefficients"].apply(lambda x: np.array(x))
    df["Nonzero-Features"] = df["Nonzero-Features"].apply(lambda x: np.array(x))
    df["Nonzero-Features-Count"] = df["Nonzero-Features"].apply(lambda x: len(x))
    df["y_pred"] = df["y_pred"].apply(lambda x: np.array(x))
    df["y_test"] = df["y_test"].apply(lambda x: np.array(x))

    # Add comparison columns.
    for metric in ["MSE", "RMSE"]:
        df[f"{metric}-Better-Than-Dummy"] = (
            df[f"Dummy-{metric}"] > df[f"{metric}"]
        )  # lower is better
        df[f"{metric}-Delta-Dummy"] = df[f"Dummy-{metric}"] - df[f"{metric}"]

    for metric in ["AUCROC", "Accuracy"]:
        df[f"{metric}-Better-Than-Dummy"] = (
            df[f"Dummy-{metric}"] < df[f"{metric}"]
        )  # higher is better
        df[f"{metric}-Delta-Dummy"] = df[f"Dummy-{metric}"] - df[f"{metric}"]

    # Sort!
    df.sort_values(["Layer", "Inputs", "Model", "Token-Pos"], inplace=True)

    return df


def get_best_layer(
    df,
    dataset_name: Optional[str] = None,
    task: str = "regression",
    metric: str = "RMSE",
    nr_rows: int = 1,
    get_values: bool = True,
    mode: str = "best",  # "best", "worst", or "median"
):
    """Finds best, worst, or median layer based on the given metric."""
    ascending_order = True if task == "regression" else False
    sorted_df = df.sort_values(
        by=["Dataset", "Layer", metric],
        ascending=[True, True, ascending_order],
    )
    grouped = sorted_df.groupby(["Dataset"])
    cols = ["Dataset", "Layer"]
    if mode == "worst":
        df_selected = grouped[cols].tail(nr_rows)
    elif mode == "median":
        df_selected = (
            grouped[cols]
            .apply(
                lambda x: x.iloc[
                    max(0, (len(x) - nr_rows) // 2) : (len(x) + nr_rows) // 2
                ]
            )
            .reset_index(drop=True)
        )
    else:  # "best"
        df_selected = grouped[cols].head(nr_rows)

    df_selected = df_selected.reset_index()
    if dataset_name is not None:
        df_selected = df_selected.loc[
            (df_selected["Dataset"].str.contains(dataset_name))
        ]
    if get_values:
        return df_selected["Layer"].iloc[0]
    return df_selected


def get_best_coefficients(
    df,
    dataset_name: Optional[str] = None,
    task: str = "regression",
    metric: str = "RMSE",
    nr_rows: int = 1,
    get_values: bool = True,
    mode: str = "best",  # "best", "worst", or "median"
):
    """Finds best, worst, or median coefficients based on the given metric."""
    ascending_order = True if task == "regression" else False
    sorted_df = df.sort_values(
        by=["Dataset", "Layer", metric],
        ascending=[True, True, ascending_order],
    )
    grouped = sorted_df.groupby(["Dataset", "Layer"])
    cols = ["Dataset", "Coefficients"]
    if mode == "worst":
        df_selected = grouped[cols].tail(nr_rows)
    elif mode == "median":
        df_selected = (
            grouped[cols]
            .apply(
                lambda x: x.iloc[
                    max(0, (len(x) - nr_rows) // 2) : (len(x) + nr_rows) // 2
                ]
            )
            .reset_index(drop=True)
        )
    else:  # "best"
        df_selected = grouped[cols].head(nr_rows)
    df_selected = df_selected.reset_index()

    if dataset_name is not None:
        df_selected = df_selected.loc[
            (df_selected["Dataset"].str.contains(dataset_name))
        ]
    if get_values:
        return df_selected.Coefficients.values

    return df_selected
