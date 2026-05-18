import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def plot_3d_surface_art(
    results_df,
    feature_name,
    penalty,
    metric="Test F1",
    smooth_factor=8,
    elev=30,
    azim=-135,
    cmap="inferno",
    figsize=(10, 7),
    save_path=None,
    transparent=False,
):
    subset = results_df[
        (results_df["Feature"] == feature_name)
        & (results_df["Penalty"].str.lower() == penalty.lower())
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No data found for feature_name={feature_name!r}, penalty={penalty!r}"
        )

    pivot_table = subset.pivot(index="max_features", columns="C", values=metric)

    y_vals = pivot_table.index.to_numpy(dtype=float)
    x_vals = pivot_table.columns.to_numpy(dtype=float)
    Z = pivot_table.to_numpy(dtype=float)

    Z_smooth = zoom(Z, smooth_factor, order=3)

    x_smooth = np.linspace(x_vals.min(), x_vals.max(), Z_smooth.shape[1])
    y_smooth = np.linspace(y_vals.min(), y_vals.max(), Z_smooth.shape[0])

    X_smooth, Y_smooth = np.meshgrid(x_smooth, y_smooth)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X_smooth,
        Y_smooth,
        Z_smooth,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    plt.tight_layout(pad=0)

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=transparent,
        )

    plt.show()


def make_surface_filename(feature_name, penalty, metric):
    if "BoW" in feature_name:
        rep = "bow"
    elif "TF-IDF" in feature_name:
        rep = "tfidf"
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")

    if "Uni+Bigram" in feature_name:
        ngram = "uni_bigram"
    elif "Bigram Only" in feature_name:
        ngram = "bigram"
    elif "Unigram" in feature_name:
        ngram = "unigram"
    else:
        raise ValueError(f"Could not parse n-gram type from: {feature_name}")

    metric_map = {
        "Test F1": "f1",
        "Test AUC": "auc",
        "F1 Gap": "gap",
    }
    metric_part = metric_map[metric]

    return f"{rep}_{penalty.lower()}_{ngram}_{metric_part}_surface.png"


def save_surface_plot(
    results_df,
    feature_name,
    penalty,
    metric="Test F1",
    smooth_factor=8,
    elev=30,
    azim=-135,
    cmap="inferno",
    figsize=(10, 7),
    save_path=None,
    transparent=False,
    show=False,
):
    subset = results_df[
        (results_df["Feature"] == feature_name)
        & (results_df["Penalty"].str.lower() == penalty.lower())
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No data found for feature_name={feature_name!r}, penalty={penalty!r}"
        )

    pivot_table = subset.pivot(index="max_features", columns="C", values=metric)

    y_vals = pivot_table.index.to_numpy(dtype=float)
    x_vals = pivot_table.columns.to_numpy(dtype=float)
    Z = pivot_table.to_numpy(dtype=float)

    Z_smooth = zoom(Z, smooth_factor, order=3)

    x_smooth = np.linspace(x_vals.min(), x_vals.max(), Z_smooth.shape[1])
    y_smooth = np.linspace(y_vals.min(), y_vals.max(), Z_smooth.shape[0])

    X_smooth, Y_smooth = np.meshgrid(x_smooth, y_smooth)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X_smooth,
        Y_smooth,
        Z_smooth,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.grid(False)

    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    plt.tight_layout(pad=0)

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=transparent,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_all_surface_plots(
    results_df,
    output_dir="surface_plots",
    metrics=("Test F1", "Test AUC", "F1 Gap"),
    penalties=("l1", "l2"),
    smooth_factor=8,
    elev=30,
    azim=-135,
    cmap="inferno",
    figsize=(10, 7),
    transparent=False,
):
    os.makedirs(output_dir, exist_ok=True)

    feature_names = sorted(results_df["Feature"].unique())

    saved_files = []

    for feature_name in feature_names:
        for penalty in penalties:
            for metric in metrics:
                filename = make_surface_filename(feature_name, penalty, metric)
                save_path = os.path.join(output_dir, filename)

                save_surface_plot(
                    results_df=results_df,
                    feature_name=feature_name,
                    penalty=penalty,
                    metric=metric,
                    smooth_factor=smooth_factor,
                    elev=elev,
                    azim=azim,
                    cmap=cmap,
                    figsize=figsize,
                    save_path=save_path,
                    transparent=transparent,
                    show=False,
                )

                saved_files.append(save_path)

    return saved_files
