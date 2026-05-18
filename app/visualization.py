import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import zoom

from config import (
    COLOR_SCALE,
    DEFAULT_CAMERA_AZIM,
    DEFAULT_CAMERA_ELEV,
    METRICS,
    NGRAM_SETTINGS,
    PENALTIES,
    REPRESENTATIONS,
)


REQUIRED_COLUMNS = {
    "Feature",
    "Penalty",
    "max_features",
    "C",
    "Test F1",
    "Test AUC",
    "F1 Gap",
}


@dataclass(frozen=True)
class LandscapeSelection:
    representation: str
    ngram: str
    penalty: str
    metric: str

    @property
    def title(self):
        return f"{self.representation} {self.ngram} | {self.penalty} | {self.metric}"


def load_results(results_path: Path):
    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find saved results CSV at {results_path}. "
            "Place the processed landscape results at this path to use the explorer."
        )

    df = pd.read_csv(results_path)
    validate_schema(df)
    return prepare_results(df)


def validate_schema(df: pd.DataFrame):
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "The results CSV is missing required columns: "
            + ", ".join(missing)
            + ". Expected at least: "
            + ", ".join(sorted(REQUIRED_COLUMNS))
        )


def prepare_results(df: pd.DataFrame):
    out = df.copy()
    out["Feature"] = out["Feature"].astype(str).str.strip()
    out["Penalty"] = out["Penalty"].astype(str).str.strip()
    out["PenaltyNorm"] = out["Penalty"].str.upper()
    out["Representation"] = out["Feature"].apply(parse_representation)
    out["Ngram"] = out["Feature"].apply(parse_ngram)

    numeric_columns = ["max_features", "C", *METRICS]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["max_features", "C", *METRICS]).copy()
    out["log10_C"] = np.log10(out["C"])
    return out


def parse_representation(feature):
    text = normalize_text(feature)
    if "tfidf" in text or "tf idf" in text:
        return "TF-IDF"
    if "bow" in text or "bag of words" in text:
        return "BoW"
    return "Unknown"


def parse_ngram(feature):
    text = normalize_text(feature)
    if "uni bigram" in text or "unibigram" in text or "uni and bigram" in text:
        return "Uni+Bigram"
    if "bigram only" in text or re.search(r"\bbigram\b", text):
        return "Bigram Only"
    if "unigram" in text:
        return "Unigram"
    return "Unknown"


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def available_landscape_summary(df):
    return (
        df[["Feature", "Representation", "Ngram", "PenaltyNorm"]]
        .drop_duplicates()
        .sort_values(["Representation", "Ngram", "PenaltyNorm", "Feature"])
    )


def validate_available_labels(df):
    unknown = df[(df["Representation"] == "Unknown") | (df["Ngram"] == "Unknown")]
    if unknown.empty:
        return None
    labels = sorted(unknown["Feature"].unique())
    return (
        "Some Feature labels could not be parsed into representation/n-gram "
        "categories. Available unparsed labels: " + ", ".join(labels)
    )


def filter_landscape(df, selection: LandscapeSelection):
    subset = df[
        (df["Representation"] == selection.representation)
        & (df["Ngram"] == selection.ngram)
        & (df["PenaltyNorm"] == selection.penalty.upper())
    ].copy()
    return subset.sort_values(["max_features", "C"])


def metric_range(df, metric, common_scale, local_df=None):
    if metric == "F1 Gap":
        values = pd.to_numeric((df if common_scale else local_df)[metric], errors="coerce").dropna()
        if values.empty:
            return 0.0, None
        return 0.0, float(values.max())
    if not common_scale:
        return None, None
    if metric in {"Test F1", "Test AUC"}:
        return 0.0, 1.0
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if values.empty:
        return None, None
    return float(values.min()), float(values.max())


def make_metric_grid(subset, metric):
    pivot = subset.pivot(index="max_features", columns="log10_C", values=metric)
    c_lookup = subset.pivot(index="max_features", columns="log10_C", values="C")
    pivot = pivot.sort_index().sort_index(axis=1)
    c_lookup = c_lookup.reindex(index=pivot.index, columns=pivot.columns)
    return pivot, c_lookup


def grid_axis_values(pivot):
    return np.arange(len(pivot.columns)), np.arange(len(pivot.index))


def axis_tick_values(length):
    if length <= 8:
        return list(range(length))
    return sorted(set([0, length // 4, length // 2, (3 * length) // 4, length - 1]))


def heatmap_tick_settings(pivot):
    x_ticks = axis_tick_values(len(pivot.columns))
    y_ticks = axis_tick_values(len(pivot.index))
    x_labels = [f"{pivot.columns[i]:.1f}" for i in x_ticks]
    y_labels = [str(int(pivot.index[i])) for i in y_ticks]
    return x_ticks, x_labels, y_ticks, y_labels


def hover_text_grid(pivot, c_lookup, selection: LandscapeSelection):
    hover = []
    for max_features in pivot.index:
        row = []
        for log_c in pivot.columns:
            value = pivot.loc[max_features, log_c]
            c_value = c_lookup.loc[max_features, log_c]
            row.append(
                f"Feature: {selection.representation} {selection.ngram}<br>"
                f"Penalty: {selection.penalty}<br>"
                f"C: {c_value:.3g}<br>"
                f"log10(C): {log_c:.2f}<br>"
                f"max_features: {int(max_features)}<br>"
                f"{selection.metric}: {value:.4f}"
            )
        hover.append(row)
    return hover


def make_heatmap(df, selection: LandscapeSelection, common_scale=False, height=760):
    subset = filter_landscape(df, selection)
    if subset.empty:
        return None, f"No rows found for {selection.title}."

    pivot, c_lookup = make_metric_grid(subset, selection.metric)
    zmin, zmax = metric_range(df, selection.metric, common_scale, subset)

    x_grid, y_grid = grid_axis_values(pivot)
    x_ticks, x_labels, y_ticks, y_labels = heatmap_tick_settings(pivot)

    fig = go.Figure(
        data=go.Heatmap(
            x=x_grid,
            y=y_grid,
            z=pivot.values,
            zmin=zmin,
            zmax=zmax,
            colorscale=COLOR_SCALE,
            colorbar={"title": selection.metric},
            text=hover_text_grid(pivot, c_lookup, selection),
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title={"text": ""},
        height=height,
        width=height,
        autosize=False,
        margin={"l": 70, "r": 30, "t": 70, "b": 70},
        xaxis_title="log10(C)",
        yaxis_title="max_features",
        font={"family": "Georgia, Times New Roman, serif", "size": 15},
        title_font={"family": "Georgia, Times New Roman, serif", "size": 24},
        xaxis={
            "tickmode": "array",
            "tickvals": x_ticks,
            "ticktext": x_labels,
            "constrain": "domain",
            "scaleanchor": "y",
            "scaleratio": 1,
            "title_font": {"size": 18},
            "tickfont": {"size": 14},
        },
        yaxis={
            "tickmode": "array",
            "tickvals": y_ticks,
            "ticktext": y_labels,
            "autorange": True,
            "constrain": "domain",
            "title_font": {"size": 18},
            "tickfont": {"size": 14},
        },
    )
    return fig, None


def camera_eye_from_elev_azim(elev=DEFAULT_CAMERA_ELEV, azim=DEFAULT_CAMERA_AZIM, radius=1.8):
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    return {
        "x": radius * np.cos(elev_rad) * np.cos(azim_rad),
        "y": radius * np.cos(elev_rad) * np.sin(azim_rad),
        "z": radius * np.sin(elev_rad),
    }


def make_surface(
    df,
    selection: LandscapeSelection,
    common_scale=False,
    smooth=False,
    show_points=True,
    height=650,
    show_title=False,
):
    subset = filter_landscape(df, selection)
    if subset.empty:
        return None, f"No rows found for {selection.title}."

    pivot, c_lookup = make_metric_grid(subset, selection.metric)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    surface_x = x
    surface_y = y
    surface_z = z

    if smooth and z.shape[0] > 1 and z.shape[1] > 1:
        surface_z = zoom(z, 3, order=3)
        surface_x = np.linspace(x.min(), x.max(), surface_z.shape[1])
        surface_y = np.linspace(y.min(), y.max(), surface_z.shape[0])

    zmin, zmax = metric_range(df, selection.metric, common_scale, subset)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=surface_z,
            colorscale=COLOR_SCALE,
            cmin=zmin,
            cmax=zmax,
            colorbar={"title": selection.metric},
            hovertemplate=(
                "log10(C): %{x:.2f}<br>"
                "max_features: %{y:.0f}<br>"
                f"{selection.metric}: " + "%{z:.4f}<extra></extra>"
            ),
        )
    )

    if show_points:
        point_hover = []
        for row in subset.to_dict("records"):
            point_hover.append(
                f"Feature: {selection.representation} {selection.ngram}<br>"
                f"Penalty: {selection.penalty}<br>"
                f"C: {row['C']:.3g}<br>"
                f"log10(C): {row['log10_C']:.2f}<br>"
                f"max_features: {int(row['max_features'])}<br>"
                f"{selection.metric}: {row[selection.metric]:.4f}"
            )
        fig.add_trace(
            go.Scatter3d(
                x=subset["log10_C"],
                y=subset["max_features"],
                z=subset[selection.metric],
                mode="markers",
                marker={"size": 2.6, "color": "white", "opacity": 0.75},
                name="Original grid points",
                text=point_hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title={"text": selection.title if show_title else ""},
        height=height,
        autosize=True,
        margin={"l": 0, "r": 0, "t": 95 if show_title else 28, "b": 0},
        font={"family": "Georgia, Times New Roman, serif", "size": 15},
        title_font={"family": "Georgia, Times New Roman, serif", "size": 24},
        uirevision="keep-camera",
        scene={
            "xaxis_title": "log10(C)",
            "yaxis_title": "max_features",
            "zaxis_title": selection.metric,
            "camera": {"eye": camera_eye_from_elev_azim()},
            "uirevision": "keep-camera",
            "aspectmode": "cube",
            "dragmode": "turntable",
            "xaxis": {"title_font": {"size": 16}, "tickfont": {"size": 12}},
            "yaxis": {"title_font": {"size": 16}, "tickfont": {"size": 12}},
            "zaxis": {"title_font": {"size": 16}, "tickfont": {"size": 12}},
        },
    )
    return fig, None


def make_comparison_heatmaps(df, left: LandscapeSelection, right: LandscapeSelection, common_scale=True):
    left_subset = filter_landscape(df, left)
    right_subset = filter_landscape(df, right)
    if left_subset.empty or right_subset.empty:
        missing = []
        if left_subset.empty:
            missing.append(left.title)
        if right_subset.empty:
            missing.append(right.title)
        return None, "No rows found for: " + "; ".join(missing)

    left_pivot, left_c = make_metric_grid(left_subset, left.metric)
    right_pivot, right_c = make_metric_grid(right_subset, right.metric)
    left_x, left_y = grid_axis_values(left_pivot)
    right_x, right_y = grid_axis_values(right_pivot)
    left_xticks, left_xlabels, left_yticks, left_ylabels = heatmap_tick_settings(left_pivot)
    right_xticks, right_xlabels, right_yticks, right_ylabels = heatmap_tick_settings(right_pivot)

    left_zmin, left_zmax = metric_range(df, left.metric, common_scale, left_subset)
    right_zmin, right_zmax = metric_range(df, right.metric, common_scale, right_subset)
    shared_coloraxis = common_scale and left.metric == right.metric

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Heatmap(
            x=left_x,
            y=left_y,
            z=left_pivot.values,
            zmin=None if shared_coloraxis else left_zmin,
            zmax=None if shared_coloraxis else left_zmax,
            coloraxis="coloraxis" if shared_coloraxis else None,
            colorscale=COLOR_SCALE,
            text=hover_text_grid(left_pivot, left_c, left),
            hoverinfo="text",
            colorbar=None if shared_coloraxis else {"title": left.metric, "x": 0.43},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=right_x,
            y=right_y,
            z=right_pivot.values,
            zmin=None if shared_coloraxis else right_zmin,
            zmax=None if shared_coloraxis else right_zmax,
            coloraxis="coloraxis" if shared_coloraxis else None,
            colorscale=COLOR_SCALE,
            text=hover_text_grid(right_pivot, right_c, right),
            hoverinfo="text",
            colorbar=None if shared_coloraxis else {"title": right.metric},
        ),
        row=1,
        col=2,
    )

    if shared_coloraxis:
        zmin, zmax = metric_range(df, left.metric, common_scale, left_subset)
        fig.update_layout(
            coloraxis={
                "colorscale": COLOR_SCALE,
                "cmin": zmin,
                "cmax": zmax,
                "colorbar": {"title": left.metric},
            }
        )

    fig.update_xaxes(
        title_text="log10(C)",
        tickmode="array",
        tickvals=left_xticks,
        ticktext=left_xlabels,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="log10(C)",
        tickmode="array",
        tickvals=right_xticks,
        ticktext=right_xlabels,
        scaleanchor="y2",
        scaleratio=1,
        constrain="domain",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="max_features",
        tickmode="array",
        tickvals=left_yticks,
        ticktext=left_ylabels,
        constrain="domain",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="max_features",
        tickmode="array",
        tickvals=right_yticks,
        ticktext=right_ylabels,
        constrain="domain",
        row=1,
        col=2,
    )
    fig.update_layout(
        height=740,
        margin={"l": 65, "r": 45, "t": 110, "b": 75},
        font={"family": "Georgia, Times New Roman, serif", "size": 15},
        title_font={"family": "Georgia, Times New Roman, serif", "size": 22},
        annotations=[],
    )
    return fig, None


def selection_from_dict(values):
    return LandscapeSelection(
        representation=values["representation"],
        ngram=values["ngram"],
        penalty=values["penalty"],
        metric=values["metric"],
    )


def validate_selection_options(df):
    parsed = {
        "representations": sorted(v for v in df["Representation"].dropna().unique() if v != "Unknown"),
        "ngrams": sorted(v for v in df["Ngram"].dropna().unique() if v != "Unknown"),
        "penalties": sorted(df["PenaltyNorm"].dropna().unique()),
    }
    expected_missing = {
        "representations": sorted(set(REPRESENTATIONS).difference(parsed["representations"])),
        "ngrams": sorted(set(NGRAM_SETTINGS).difference(parsed["ngrams"])),
        "penalties": sorted(set(PENALTIES).difference(parsed["penalties"])),
    }
    return parsed, expected_missing
