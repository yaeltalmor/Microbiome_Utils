import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
from matplotlib.lines import Line2D

def create_metric_dict(metric_name: str, thresholds_list : list, all_versions_metrics_dict: dict):
    '''

    :param metric_name: "pr_auc","roc_auc","accuracy","f1","precision","recall"
    :param thresholds_list: list desired thresholds, e.g. [24,36,48,60]
    :param all_versions_metrics_dict:
    {24: [{'accuracy': 0.5915492957746479,
    'f1': 0.3378995433789954,
    'roc_auc': 0.6724949958789591,
    'precision': 0.22839506172839505,
    'recall': 0.6491228070175439,
    'pr_auc': 0.24208953894355623},{},...],36: [....],...}
    :return: dict of dicts for each model and threshold with list of metric values across folds
    '''
    metric_dict = {}
    for model_name, model_metrics in all_versions_metrics_dict.items():
        metric_dict[model_name]={i: [fold_metrics[metric_name] for fold_metrics in model_metrics[i]] for i in thresholds_list}
    return metric_dict

def metric_dict_to_dataframe(metric_dict, metric_name):
    """
    Converts nested metric_dict[model][threshold] = list of values
    → to long DataFrame with columns ['Model', 'Threshold', 'Value'].
    """
    records = []
    for model_name, threshold_dict in metric_dict.items():
        for threshold, values_list in threshold_dict.items():
            for v in values_list:
                records.append({
                    "model": model_name,
                    "threshold": threshold,
                    f"{metric_name}": v
                })
    return pd.DataFrame(records)

def augment_aupr_df(aupr_df, drug_switch_targets):
    aupr_df["pos_rate"] = aupr_df.apply(
        lambda row: drug_switch_targets[row["threshold"]].mean(), axis=1
    )
    aupr_df["aupr_diff"] = aupr_df["AUPR"] - aupr_df["pos_rate"]
    aupr_df["aupr_diff_to_neg"] = aupr_df["aupr_diff"] / (1 - aupr_df["pos_rate"])
    return aupr_df


def summerize_results_per_category(model_full_results_df: pd.DataFrame):

    # Group by observation window (Threshold)
    by_threshold = model_full_results_df.groupby("threshold")[["AUC", "aupr_diff_to_neg", "AUPR"]].mean()
    best_threshold_auc = by_threshold["AUC"].idxmax()
    best_threshold_aupr_diff_to_neg = by_threshold["aupr_diff_to_neg"].idxmax()
    best_threshold_ap = by_threshold["AUPR"].idxmax()
    print("Average Performance by Threshold:")
    print(by_threshold)
    print("\nBest AUC Threshold:", best_threshold_auc)
    print("Best aupr_diff_to_neg Threshold:", best_threshold_aupr_diff_to_neg)
    print("Best AUPR Threshold:", best_threshold_ap)
    # Group by model variant (Source)
    by_model = model_full_results_df.groupby("model")[["AUC", "aupr_diff_to_neg", "AUPR"]].mean()
    best_model_auc = by_model["AUC"].idxmax()
    best_model_aupr_diff_to_neg = by_model["aupr_diff_to_neg"].idxmax()
    best_model_ap = by_model["AUPR"].idxmax()
    print("\nAverage Performance by Model Variant:")
    print(by_model)
    print("\nBest AUC Model:", best_model_auc)
    print("Best aupr_diff_to_neg Model:", best_model_aupr_diff_to_neg)
    print("Best AUPR Model:", best_model_ap)
    by_model_thres = model_full_results_df.groupby(["model", "threshold"])[["AUC", "aupr_diff_to_neg", "AUPR"]].mean()
    best_model_version_auc = by_model_thres["AUC"].idxmax()
    best_model_version_aupr_diff_to_neg = by_model_thres["aupr_diff_to_neg"].idxmax()
    best_model_version_ap = by_model_thres["AUPR"].idxmax()
    print("\nBest Performing Model Version:")
    print(by_model_thres)
    print(f"\nBest AUC Model Version: {best_model_version_auc}, with AUC of: {by_model_thres["AUC"].max()}")
    print(
        f"Best aupr_diff_to_neg Model Version:, {best_model_version_aupr_diff_to_neg}, with aupr_diff_to_neg of: {by_model_thres["aupr_diff_to_neg"].max()}")
    print(f"Best AUPR Model Version:, {best_model_version_ap}, with AUPR of: {by_model_thres["AUPR"].max()}")


def plot_metric_distributions(
    df1,
    cols_to_plot,
    stratify_by,
    df2=None,
    df1_label=None,
    df2_label=None,
    fig_name=None,
    color1="#35978f",
    color2="#01665e",
    alpha=0.4,
    line_alpha=0.3,
    box_alpha=0.9,
    figsize=(12, 5),
    color_palette="BrBG",
    font_scale=1.85
):
    sns.set(style="whitegrid", font_scale=font_scale)
    compare = df2 is not None
    ncols = len(cols_to_plot)

    fig, axes = plt.subplots(
        1, ncols,
        figsize=(figsize[0] * ncols / 2, figsize[1]),
        sharey=False
    )
    if ncols == 1:
        axes = [axes]

    # Parse stratification
    if isinstance(stratify_by, list) and len(stratify_by) == 2:
        x_col, hue_col = stratify_by
    else:
        x_col = stratify_by
        hue_col = None

    # --- Define hue levels & palette ONCE (for single-df mode) ---
    if not compare and hue_col:
        # Use appearance order (or categorical order) so we can
        # feed the same to seaborn via hue_order
        hue_levels = list(df1[hue_col].unique())
        palette_list = sns.color_palette(color_palette, len(hue_levels))
    else:
        hue_levels = [None]
        palette_list = [color1]

    for idx, (ax, metric) in enumerate(zip(axes, cols_to_plot)):

        if compare:
            # --- Prepare data for paired mode ---
            df1_plot = df1.copy().reset_index(drop=True)
            df2_plot = df2.copy().reset_index(drop=True)
            if len(df1_plot) != len(df2_plot):
                raise ValueError("DataFrames must have the same length for paired plotting.")

            combined = pd.concat([
                df1_plot.assign(version=df1_label),
                df2_plot.assign(version=df2_label)
            ])

            # Combine hue variables for palette consistency
            if hue_col:
                combined["hue_combo"] = combined[hue_col] + " | " + combined["version"]
                models = sorted(combined[hue_col].unique())
                palette = {f"{m} | {df1_label}": color1 for m in models}
                palette.update({f"{m} | {df2_label}": color2 for m in models})
            else:
                combined["hue_combo"] = combined["version"]
                palette = {df1_label: color1, df2_label: color2}

            # --- Compute offsets for plotting ---
            x_levels = sorted(df1_plot[x_col].unique())
            local_hue_levels = sorted(df1_plot[hue_col].unique()) if hue_col else [None]
            hue_offsets = np.linspace(-0.25, 0.25, len(local_hue_levels)) if hue_col else [0]

            # --- First: Paprika connecting lines (behind everything) ---
            for i, x_val in enumerate(x_levels):
                for j, h_val in enumerate(local_hue_levels):
                    mask1 = (df1_plot[x_col] == x_val) & ((df1_plot[hue_col] == h_val) if hue_col else True)
                    mask2 = (df2_plot[x_col] == x_val) & ((df2_plot[hue_col] == h_val) if hue_col else True)
                    sub1 = df1_plot.loc[mask1, metric].reset_index(drop=True)
                    sub2 = df2_plot.loc[mask2, metric].reset_index(drop=True)
                    n = min(len(sub1), len(sub2))
                    sub1, sub2 = sub1[:n], sub2[:n]

                    x1 = np.full(n, i + hue_offsets[j] - 0.15)
                    x2 = np.full(n, i + hue_offsets[j] + 0.15)
                    color_line = "#80cdc1"
                    for xi1, xi2, y1, y2 in zip(x1, x2, sub1, sub2):
                        ax.plot([xi1, xi2], [y1, y2], color=color_line, alpha=line_alpha, zorder=1)

            # --- Second: Scatter points ---
            for i, x_val in enumerate(x_levels):
                for j, h_val in enumerate(local_hue_levels):
                    mask1 = (df1_plot[x_col] == x_val) & ((df1_plot[hue_col] == h_val) if hue_col else True)
                    mask2 = (df2_plot[x_col] == x_val) & ((df2_plot[hue_col] == h_val) if hue_col else True)
                    sub1 = df1_plot.loc[mask1, metric]
                    sub2 = df2_plot.loc[mask2, metric]
                    x1 = np.full(len(sub1), i + hue_offsets[j] - 0.15)
                    x2 = np.full(len(sub2), i + hue_offsets[j] + 0.15)
                    ax.scatter(x1, sub1, color=color1, alpha=alpha, zorder=2)
                    ax.scatter(x2, sub2, color=color2, alpha=alpha, zorder=2)

            # --- Third: Boxplot on top ---
            box = sns.boxplot(
                data=combined,
                x=x_col,
                y=metric,
                hue="hue_combo",
                ax=ax,
                showcaps=True,
                fliersize=0,
                linewidth=1,
                palette=palette,
                boxprops=dict(alpha=box_alpha),
                width=0.7
            )
            # ax.legend_.remove()

            # Ensure boxes appear above all points/lines
            for patch in box.artists:
                patch.set_zorder(4)

            # --- Custom legend ---
            if idx == 0:  # show legend only on second subplot

                custom_lines = [
                    Line2D([0], [0], marker='o', color='w', label=df1_label, markerfacecolor=color1, markersize=9),
                    Line2D([0], [0], marker='o', color='w', label=df2_label, markerfacecolor=color2, markersize=9)
                ]
                ax.legend(handles=custom_lines, title="", loc="best", frameon=True)

        else:
            # --- Single dataframe mode ---
            x_levels = sorted(df1[x_col].unique())
            # use the global hue_levels & palette_list we computed
            local_hue_levels = hue_levels
            hue_offsets = np.linspace(-0.2, 0.2, len(local_hue_levels)) if hue_col else [0]

            # Scatter first (so dots are behind)
            for i, x_val in enumerate(x_levels):
                for j, h_val in enumerate(local_hue_levels):
                    mask = (df1[x_col] == x_val) & ((df1[hue_col] == h_val) if hue_col else True)
                    sub = df1.loc[mask, metric]
                    x = np.full(len(sub), i + hue_offsets[j])
                    ax.scatter(
                        x, sub,
                        alpha=alpha,
                        color=palette_list[j] if hue_col else color1,
                        zorder=1  # behind box
                    )

            # Boxplot on top, using SAME palette & hue order
            box = sns.boxplot(
                data=df1,
                x=x_col,
                y=metric,
                hue=hue_col,
                hue_order=local_hue_levels if hue_col else None,
                ax=ax,
                fliersize=0,
                linewidth=1,
                boxprops=dict(alpha=box_alpha),
                width=0.6,
                palette=palette_list if hue_col else [color1]
            )

            for patch in box.artists:
                patch.set_zorder(3)

        # Remove per-axis legend (we’ll handle shared legend later)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric)

    # --------------------------------------------------------------------
    # LEGENDS
    # --------------------------------------------------------------------

    if compare:
        # (compare legend already added inside first subplot)
        first_ax = axes[0]
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color1,
                   markersize=10, label=df1_label),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color2,
                   markersize=10, label=df2_label)
        ]
        first_ax.legend(handles=handles, title="", loc="best", frameon=True)
    else:
        # SINGLE-DF MODE: shared external legend
        if hue_col:
            handles = [
                Line2D(
                    [0], [0],
                    marker='s',
                    color='w',
                    markerfacecolor=palette_list[i],
                    markersize=12,
                    label=str(hue_levels[i])
                )
                for i in range(len(hue_levels))
            ]

            fig.subplots_adjust(right=0.88)

            fig.legend(
                handles,
                [h.get_label() for h in handles],
                title=hue_col,
                loc="center left",
                bbox_to_anchor=(0.84, 0.5),
                frameon=True
            )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if fig_name:
        plt.savefig(f"{fig_name}.png", dpi=300, bbox_inches="tight")

    plt.show()






