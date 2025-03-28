import pandas as pd
import numpy as np
from typing import Optional


def postprocess_df_probes(
    df, filter_error_type: Optional[str] = "sm", filter_probe_token_pos: Optional[str] = "Last", filter_inputs: Optional[str] = "Activations"
) -> pd.DataFrame:
    """Process the probe performance result per LM model and apply filters at the end."""

    df["Layer"] = df["Layer"].astype(int)
    df["Model"] = df["Model"].str.replace("Lasso-", "L-", regex=False)
    df["Residuals"] = df["Residuals"].apply(lambda x: np.array(x))
    df["Coefficients"] = df["Coefficients"].apply(lambda x: np.array(x))
    df["Nonzero-Features"] = df["Nonzero-Features"].apply(lambda x: np.array(x))
    df["Nonzero-Features-Count"] = df["Nonzero-Features"].apply(lambda x: len(x))
    df["y_pred"] = df["y_pred"].apply(lambda x: np.array(x))
    df["y_test"] = df["y_test"].apply(lambda x: np.array(x))

    # Add compariso columns.
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

    df.sort_values(["Layer", "Inputs", "Model", "Match-Type"], inplace=True)

    df["Dataset_name"] = df["Dataset"].apply(
        lambda x: (
            "MMLU"
            if "mmlu" in x.lower()
            else (
                "SMS SPAM"
                if "sms" in x.lower()
                else (
                    "Sentiment"
                    if "sentiment" in x.lower()
                    else "Yes_No" if "yes_no" in x.lower() else "Unknown"
                )
            )
        )
    )
    
    if filter_error_type is not None:
        df = df.loc[df["Error-Type"] == filter_error_type]
    
    if filter_inputs is not None:
         df = df.loc[df["Inputs"] == filter_inputs]

    if filter_probe_token_pos is not None:
        df = df.loc[df["Match-Type"] == filter_probe_token_pos]
    
    return df


def get_best_layer(
    df,
    task_name: str,
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
        df_selected = grouped[cols].apply(lambda x: x.iloc[max(0, (len(x) - nr_rows) // 2):(len(x) + nr_rows) // 2]).reset_index(drop=True)
    else:  # "best"
        df_selected = grouped[cols].head(nr_rows)
    
    df_selected = df_selected.reset_index()
    if get_values:
        return df_selected.loc[
            (df_selected["Dataset"].str.contains(task_name)), "Layer"
        ].iloc[0]
    return df_selected

def get_best_coefficients(
    df,
    task_name: str,
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
        df_selected = grouped[cols].apply(lambda x: x.iloc[max(0, (len(x) - nr_rows) // 2):(len(x) + nr_rows) // 2]).reset_index(drop=True)
    else:  # "best"
        df_selected = grouped[cols].head(nr_rows)    
    df_selected = df_selected.reset_index()
    if get_values:
        return df_selected[
            (df_selected["Dataset"].str.contains(task_name))
        ].Coefficients.values
    
    return df_selected

