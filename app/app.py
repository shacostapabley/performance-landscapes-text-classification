import streamlit as st

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    METRICS,
    NGRAM_SETTINGS,
    PENALTIES,
    PLOTLY_CONFIG,
    PRESETS,
    REPORT_PATH,
    REPRESENTATIONS,
    RESULTS_PATH,
)
from visualization import (
    LandscapeSelection,
    available_landscape_summary,
    load_results,
    make_heatmap,
    make_surface,
    selection_from_dict,
    validate_available_labels,
    validate_selection_options,
)


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_results():
    return load_results(RESULTS_PATH)


def render_header():
    st.markdown(
        """
        <style>
        html, body, .stApp {
            font-family: Georgia, "Times New Roman", Times, serif !important;
        }
        .stMarkdown, .stText, .stCaption, .stDataFrame, label, p, h1, h2, h3 {
            font-family: Georgia, "Times New Roman", Times, serif !important;
        }
        h1 {
            font-size: 3rem !important;
            letter-spacing: 0 !important;
        }
        h2, h3 {
            letter-spacing: 0 !important;
        }
        div[data-testid="stCaptionContainer"] {
            font-size: 1.05rem !important;
        }
        .block-container {
            padding-top: 2.2rem;
            max-width: 1500px;
        }
        .stPlotlyChart {
            margin-top: 0.5rem;
        }
        .plot-note {
            color: #555;
            font-size: 0.98rem;
            line-height: 1.45;
            border-top: 1px solid rgba(0,0,0,0.12);
            margin-top: 0.8rem;
            padding-top: 0.7rem;
        }
        .comparison-panel {
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 8px;
            padding: 1rem 1.1rem 0.85rem 1.1rem;
            background: rgba(250,250,248,0.7);
        }
        .comparison-label {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .chart-title {
            text-align: center;
            font-family: Georgia, "Times New Roman", Times, serif !important;
            font-size: 1.55rem;
            font-weight: 700;
            margin: 1.15rem 0 0.15rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.markdown(
        "Explore how logistic regression behaves across the parameter space for "
        "title-based fake news classification. The landscapes show where performance "
        "is strong, where it collapses, and where generalization becomes less stable."
    )
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "rb") as report_file:
            st.download_button(
                "Download full research report PDF",
                data=report_file,
                file_name="TextClassificationLandscapes_Report.pdf",
                mime="application/pdf",
            )
    else:
        st.caption("Full research report PDF expected at `report/final_report.pdf`.")


def centered_plotly_chart(fig, title=None, width_ratio=4, key=None):
    left, center, right = st.columns([1, width_ratio, 1])
    with center:
        if title:
            chart_title(title)
        st.plotly_chart(fig, use_container_width=False, config=PLOTLY_CONFIG, key=key)


def chart_title(text):
    st.markdown(
        f'<div class="chart-title">{text}</div>',
        unsafe_allow_html=True,
    )


def render_visual_notes(include_smoothing=True):
    smoothing = (
        " Smoothed 3D surfaces are visual interpolations between observed grid points."
        if include_smoothing
        else ""
    )
    st.markdown(
        (
            '<div class="plot-note">'
            "Plots use <code>log10(C)</code> on the x-axis because C was sampled "
            "on a log scale. Hover labels show the original C value."
            f"{smoothing}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def load_or_stop():
    try:
        df = cached_results()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    label_warning = validate_available_labels(df)
    if label_warning:
        st.warning(label_warning)
        with st.expander("Available Feature labels"):
            st.dataframe(available_landscape_summary(df), use_container_width=True)

    parsed, missing = validate_selection_options(df)
    missing_bits = [f"{key}: {', '.join(values)}" for key, values in missing.items() if values]
    if missing_bits:
        st.info(
            "Some expected categories were not found in the CSV. Available parsed "
            "categories will still be used where possible. Missing: "
            + "; ".join(missing_bits)
        )

    return df


def selection_controls(prefix, defaults=None):
    defaults = defaults or {}
    cols = st.columns(4)
    with cols[0]:
        representation = st.selectbox(
            "Representation",
            REPRESENTATIONS,
            index=REPRESENTATIONS.index(defaults.get("representation", "BoW")),
            key=f"{prefix}_representation",
        )
    with cols[1]:
        ngram = st.selectbox(
            "N-gram setting",
            NGRAM_SETTINGS,
            index=NGRAM_SETTINGS.index(defaults.get("ngram", "Unigram")),
            key=f"{prefix}_ngram",
        )
    with cols[2]:
        penalty = st.selectbox(
            "Penalty",
            PENALTIES,
            index=PENALTIES.index(defaults.get("penalty", "L2")),
            key=f"{prefix}_penalty",
        )
    with cols[3]:
        metric = st.selectbox(
            "Metric",
            METRICS,
            index=METRICS.index(defaults.get("metric", "Test F1")),
            key=f"{prefix}_metric",
        )

    return LandscapeSelection(representation, ngram, penalty, metric)


def render_plot_error(message):
    if message:
        st.error(message)
        st.caption(
            "Check whether the selected Feature/Penalty combination exists in "
            "`data/processed/landscape_results.csv`."
        )


def render_single_landscape(df, selection=None, key_prefix="single"):
    if selection is None:
        selection = selection_controls(key_prefix)

    opts = st.columns(3)
    with opts[0]:
        common_scale = st.toggle(
            "Common metric scale",
            value=False,
            key=f"{key_prefix}_common_scale",
            help=(
                "For F1 and AUC this uses 0 to 1. For F1 Gap this uses the "
                "observed range."
            ),
        )
    with opts[1]:
        smooth_surface = st.toggle(
            "Smooth surface",
            value=True,
            key=f"{key_prefix}_smooth",
            help="Interpolate between observed grid points for a smoother visual surface.",
        )
    with opts[2]:
        show_points = st.toggle(
            "Grid points",
            value=False,
            key=f"{key_prefix}_points",
        )

    heatmap, heatmap_error = make_heatmap(df, selection, common_scale=common_scale)
    if heatmap_error:
        render_plot_error(heatmap_error)
        return

    centered_plotly_chart(heatmap, title=selection.title, key=f"{key_prefix}_heatmap")

    surface, surface_error = make_surface(
        df,
        selection,
        common_scale=common_scale,
        smooth=smooth_surface,
        show_points=show_points,
    )
    if surface_error:
        render_plot_error(surface_error)
        return
    chart_title(selection.title)
    st.plotly_chart(surface, use_container_width=True, config=PLOTLY_CONFIG, key=f"{key_prefix}_surface")
    render_visual_notes(include_smoothing=True)


def render_comparison(df, left=None, right=None, key_prefix="compare"):
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="comparison-label">Left Landscape</div>', unsafe_allow_html=True)
            left_selection = left or selection_controls(
                f"{key_prefix}_left",
                {
                    "representation": "BoW",
                    "ngram": "Unigram",
                    "penalty": "L1",
                    "metric": "Test F1",
                },
            )
    with col_right:
        with st.container(border=True):
            st.markdown('<div class="comparison-label">Right Landscape</div>', unsafe_allow_html=True)
            right_selection = right or selection_controls(
                f"{key_prefix}_right",
                {
                    "representation": "BoW",
                    "ngram": "Unigram",
                    "penalty": "L2",
                    "metric": "Test F1",
                },
            )

    common_scale = st.toggle(
        "Common metric scale",
        value=False,
        key=f"{key_prefix}_common",
    )
    surface_cols = st.columns(2, gap="large")
    with surface_cols[0]:
        show_surfaces = st.toggle(
            "Show 3D surfaces",
            value=False,
            key=f"{key_prefix}_surfaces",
        )
    with surface_cols[1]:
        show_surface_points = st.toggle(
            "Surface grid points",
            value=False,
            key=f"{key_prefix}_surface_points",
            disabled=not show_surfaces,
        )

    heat_left, err_left_heat = make_heatmap(
        df,
        left_selection,
        common_scale=common_scale,
        height=560,
    )
    heat_right, err_right_heat = make_heatmap(
        df,
        right_selection,
        common_scale=common_scale,
        height=560,
    )

    heat_cols = st.columns(2, gap="large")
    with heat_cols[0]:
        if err_left_heat:
            render_plot_error(err_left_heat)
        else:
            chart_title(left_selection.title)
            st.plotly_chart(
                heat_left,
                use_container_width=False,
                config=PLOTLY_CONFIG,
                key=f"{key_prefix}_heatmap_left",
            )
    with heat_cols[1]:
        if err_right_heat:
            render_plot_error(err_right_heat)
        else:
            chart_title(right_selection.title)
            st.plotly_chart(
                heat_right,
                use_container_width=False,
                config=PLOTLY_CONFIG,
                key=f"{key_prefix}_heatmap_right",
            )

    if show_surfaces:
        surf_left, err_left = make_surface(
            df,
            left_selection,
            common_scale=common_scale,
            smooth=True,
            show_points=show_surface_points,
            height=610,
            show_title=False,
        )
        surf_right, err_right = make_surface(
            df,
            right_selection,
            common_scale=common_scale,
            smooth=True,
            show_points=show_surface_points,
            height=610,
            show_title=False,
        )
        surf_cols = st.columns(2, gap="large")
        with surf_cols[0]:
            if err_left:
                render_plot_error(err_left)
            else:
                chart_title(left_selection.title)
                st.plotly_chart(
                    surf_left,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                    key=f"{key_prefix}_surface_left",
                )
        with surf_cols[1]:
            if err_right:
                render_plot_error(err_right)
            else:
                chart_title(right_selection.title)
                st.plotly_chart(
                    surf_right,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                    key=f"{key_prefix}_surface_right",
                )
        render_visual_notes(include_smoothing=True)
    else:
        render_visual_notes(include_smoothing=False)


def render_presets(df):
    preset_name = st.selectbox("Preset finding", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    st.markdown(f"**Interpretation:** {preset['interpretation']}")

    if preset["mode"] == "single":
        render_single_landscape(
            df,
            selection=selection_from_dict(preset["left"]),
            key_prefix=f"preset_{preset_name}",
        )
    else:
        render_comparison(
            df,
            left=selection_from_dict(preset["left"]),
            right=selection_from_dict(preset["right"]),
            key_prefix=f"preset_{preset_name}",
        )


def render_notes(df):
    st.subheader("About The Landscapes")
    st.write(
        "The goal is not only to identify one best setting, but to see how feature "
        "representation, regularization, and feature budget shape model behavior. "
        "The heatmaps and surfaces make regimes, plateaus, collapse zones, and "
        "fragile regions visible."
    )

    with st.expander("What is C?"):
        st.write(
            "Smaller C means stronger regularization; larger C means weaker regularization."
        )
    with st.expander("What is max_features?"):
        st.write("Maximum number of retained vocabulary features.")
    with st.expander("What is F1?"):
        st.write(
            "F1 measures final classification performance by balancing precision and recall."
        )
    with st.expander("What is AUC?"):
        st.write(
            "AUC measures how well the model ranks fake titles above real titles overall."
        )
    with st.expander("What is F1 Gap?"):
        st.write(
            "F1 Gap is Train F1 minus Test F1; larger values suggest more overfitting "
            "or weaker generalization."
        )
    with st.expander("Why use heatmaps?"):
        st.write(
            "Heatmaps reveal regimes, plateaus, collapse zones, and fragile regions "
            "that a single best score would miss."
        )
    with st.expander("Data loaded by the app"):
        st.write(f"Results CSV: `{RESULTS_PATH}`")
        st.write(f"Rows: `{len(df):,}`")
        st.dataframe(available_landscape_summary(df), use_container_width=True)


def main():
    render_header()
    df = load_or_stop()

    tabs = st.tabs(
        [
            "Single Landscape Explorer",
            "Comparison Mode",
            "Finding Presets",
            "About / Interpretation Notes",
        ]
    )
    with tabs[0]:
        render_single_landscape(df)
    with tabs[1]:
        render_comparison(df)
    with tabs[2]:
        render_presets(df)
    with tabs[3]:
        render_notes(df)


if __name__ == "__main__":
    main()
