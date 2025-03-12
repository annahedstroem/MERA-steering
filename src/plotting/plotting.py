
from typing import Optional, Tuple, List, Dict, Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_best_coefficients_plotting(df, task_name: Optional[str] = None, task: str = "regression", metric: str = "RMSE", nr_rows: int = 1, select: bool = True):
    # Sort and find best coefficients data
    
    if task == "regression":
        metric_sorting = False if "Dummy" in metric else True
        sorted_df = df.sort_values(by=[metric, "Dataset", "Layer", "Inputs", "Match-Type"], ascending=[metric_sorting, True,  True, True, False])
    else:
        sorted_df = df.sort_values(by=[metric, "Dataset", "Layer", "Inputs", "Match-Type"], ascending=[False, True, True, True, False])
        
    df_best_coefficients = sorted_df.groupby(["Dataset", "Layer", "Inputs", "Match-Type"])[
        ["Coefficients", "Dataset", "Model", "Layer", "Inputs", "Match-Type", "Nonzero-Features-Count", metric, f"{metric}-Delta-Dummy"]
    ].head(nr_rows)
    df_best_coefficients = df_best_coefficients.reset_index()

    if select:
        assert task_name is not None, "Task name cannot be None if select is True"
        return df_best_coefficients[(df_best_coefficients["Dataset"].str.contains(task_name)) & (df_best_coefficients["Inputs"] == "Activations")].Coefficients.values

    return df_best_coefficients

def rename_group(s):
    return s.replace('gemma-2-2b', 'Gemma-2-2B').replace('it', 'IT').replace('Qwen2.5', 'Qwen-2.5').replace('-INSTRUCT', '-IT').replace('-Instruct', '-IT').replace('_', ' ').upper().replace(' HIGH SCHOOL', '-HS').replace('YES NO QUESTION', 'YES/NO').replace('SENTIMENT ANALYSIS', 'SENTIMENT')

# TODO: Update this function with filter_inputs

def plot_probe_performance_by_model_family(
    task_types: Dict[str, str],
    filename_pairs: List[Tuple[str, str]],
    error_type: str,
    nr_rows: int,
    rename_group: Callable
) -> None:
    """Plots probe performance overview across different models and tasks."""
    PLOT_SAEs = False
    nr_plots = 4
    
    for task, metric in task_types.items():
        fig, axes = plt.subplots(1, nr_plots, figsize=(2.5 * nr_plots, 2))
        dfs_family = []
        index = 0
        
        for ix, (filename, model_name) in enumerate(filename_pairs):
            PLOT_SAEs = ix in [4, 5]
            col = "Match-Type" if not PLOT_SAEs else "Inputs"
            
            df = post_process_df(pd.read_pickle(f"../runs/probes/{filename}.pkl"), filter_error_type=error_type, filter_probe_match_type="Last")
            df_exact = post_process_df(pd.read_pickle(f"../runs/probes/{filename}.pkl"), filter_error_type=error_type, filter_probe_match_type="Exact")
            df_combined = pd.concat([df, df_exact])
            
            df_best_coefficients = get_best_coefficients_plotting(
                df_combined, task_name=None, task=task, metric=metric, nr_rows=nr_rows, select=False
            )
            
            if not PLOT_SAEs:
                df_best_coefficients = df_best_coefficients.loc[df_best_coefficients["Inputs"] == "Activations"]
            dfs_family.append(df_best_coefficients)
            
            if (ix + 1) % 2 == 0:
                df_family = pd.concat(dfs_family)
                dfs_family = []
                
                df_family["Token position"] = df_family["Match-Type"].replace(['Last', 'Exact'], ['Last (prompt)', 'Exact (generation)'])
                if "Inputs" in df_family:
                    col = "Representation"
                    df_family["Representation"] = df_family["Inputs"].replace(['Activations', 'Encodings'], ['Activations', 'Sparse rep.'])
                
                print(f"Plotting at index: {index}")
                
                sns.lineplot(
                    data=df_family,
                    x="Layer",
                    y=metric,
                    hue=col,
                    style='Token position',
                    markers=False,
                    ax=axes[index],
                    legend=False if not PLOT_SAEs else "full",
                    errorbar="se",
                    palette=["black", "black"] if not PLOT_SAEs else ['#1f77b4', '#2ca02c', '#1f77b4', '#2ca02c']
                )
                
                axes[index].set_title(rename_group(model_name))
                axes[index].set_xlabel("Layer")
                axes[index].set_ylabel(metric)
                axes[index].grid(True)
                
                if PLOT_SAEs:
                    handles, labels = axes[index].get_legend_handles_labels()
                    axes[index].legend_.remove()
                index += 1
                
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25), frameon=True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        save_path_png = f"../runs/plots/probe-{metric}-averaged_{nr_rows}.png"
        save_path_svg = f"../runs/plots/probe-{metric}-averaged_{nr_rows}.svg"
        plt.savefig(save_path_png)
        plt.savefig(save_path_svg)
        print(f"Saved PNG: {save_path_png}")
        print(f"Saved SVG: {save_path_svg}")
        plt.show()

def plot_probe_performance_by_dataset(
    task_types: dict[str, str],
    filename_pairs: list[tuple[str, str]],
    error_type: str,
    nr_rows: int,
    task_names: list[str],
    rename_group: callable,
    PLOT_LEGEND: bool = False
) -> None:
    """Plots probe performance per dataset across different models."""
    for task, metric in task_types.items():
        for y_axis in [f"{metric}"]:
            for filename, model_name in filename_pairs:
                
                df = post_process_df(pd.read_pickle(f"../runs/probes/{filename}.pkl"), filter_error_type=error_type, filter_probe_match_type="Last")
                df_exact = post_process_df(pd.read_pickle(f"../runs/probes/{filename}.pkl"), filter_error_type=error_type, filter_probe_match_type="Exact")
                df_combined = pd.concat([df, df_exact])
                
                fig, axes = plt.subplots(1, len(task_names), figsize=(2.5 * len(task_names), 2), sharey=False)
                if len(task_names) == 1:
                    axes = [axes]
                
                for index, (ax, dataset) in enumerate(zip(axes, task_names)):
                    df_filtered = df_combined[df_combined["Dataset"] == dataset]
                    df_best_coefficients = get_best_coefficients_plotting(
                        df_filtered, task_name=dataset, task=task, metric=metric, nr_rows=nr_rows, select=False
                    )
                    
                    df_best_coefficients["Token position"] = df_best_coefficients["Match-Type"].replace(['Last', 'Exact'], ['Last (prompt)', 'Exact (generation)'])
                    if "Inputs" in df_best_coefficients:
                        df_best_coefficients["Features"] = df_best_coefficients["Inputs"].replace(['Activations', 'Encodings'], ['Activations', 'Sparse rep.'])
                    
                    sns.lineplot(
                        data=df_best_coefficients,
                        x="Layer",
                        y=y_axis,
                        hue="Features",
                        style="Token position", 
                        palette=['#1f77b4', '#2ca02c'] if "gemma" in model_name.lower() else ['#1f77b4'], 
                        dashes={"Last (prompt)": (1, 0), "Exact (generation)": (5, 2)},
                        markers=False, 
                        ax=ax,
                        legend=False, #"full", 
                        errorbar="sd",
                    )
                    
                    ax.set_title(f"{rename_group(dataset)}\n{rename_group(model_name)}") 
                    ax.set_xlabel("Layer")
                    ax.set_ylabel(y_axis)
                    ax.grid(True)
                    
                    #if index + 1 == len(task_names):
                    #    handles, labels = ax.get_legend_handles_labels()
                    #ax.legend_.remove()
                
                if PLOT_LEGEND:
                    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.25), frameon=True)
                plt.tight_layout()
                
                save_path_png = f"../runs/plots/probe-{model_name}-{metric}_{nr_rows}.png"
                save_path_svg = f"../runs/plots/probe-{model_name}-{metric}_{nr_rows}.svg"
                plt.savefig(save_path_png)
                plt.savefig(save_path_svg)
                plt.show()
