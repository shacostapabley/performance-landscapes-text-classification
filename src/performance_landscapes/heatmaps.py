import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(
    results_df,
    feature_name,
    penalty,
    metric="Test F1",
    smooth=False,
    cmap="inferno",
    figsize=(10, 8),
    annot=False,
    fmt=".3f",
    vmin=None,
    vmax=None,
    font_family="STIXGeneral",
    title_size=18,
    label_size=13,
    tick_size=11,
    cbar_label_size=12,
    grid=False,
):
    subset = results_df[
        (results_df["Feature"] == feature_name)
        & (results_df["Penalty"].str.lower() == penalty.lower())
    ].copy()

    pivot_table = subset.pivot(index="max_features", columns="C", values=metric)

    if pivot_table.empty:
        raise ValueError(
            f"No data found for feature_name={feature_name!r} and penalty={penalty!r}"
        )

    plt.rcParams["font.family"] = font_family

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        pivot_table.values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="bilinear" if smooth else "nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_xticklabels(
        [f"{c:.3g}" for c in pivot_table.columns],
        rotation=45,
        ha="right",
        fontsize=tick_size,
    )

    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index, fontsize=tick_size)

    ax.set_xlabel("C", fontsize=label_size, labelpad=10)
    ax.set_ylabel("max_features", fontsize=label_size, labelpad=10)
    ax.set_title(
        f"{feature_name}  |  {penalty}  |  {metric}",
        fontsize=title_size,
        pad=16,
        weight="semibold",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if grid:
        ax.set_xticks(np.arange(-0.5, len(pivot_table.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot_table.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    if annot:
        values = pivot_table.values
        local_vmin = np.nanmin(values) if vmin is None else vmin
        local_vmax = np.nanmax(values) if vmax is None else vmax
        midpoint = (local_vmin + local_vmax) / 2

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isnan(val):
                    continue
                text_color = "white" if val < midpoint else "black"
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=tick_size - 1,
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label(metric, fontsize=cbar_label_size, labelpad=10)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.show()

