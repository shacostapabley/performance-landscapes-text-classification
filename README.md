# Performance Landscapes in Text Classification

This repository contains a statistical learning project on title-based fake-news classification. The project studies logistic regression as a performance landscape shaped by feature representation, n-gram choice, regularization, and feature budget.

The central research question is: how do Bag-of-Words versus TF-IDF, unigram versus bigram feature spaces, L1 versus L2 regularization, `C`, and `max_features` affect Test F1, Test AUC, and train-test F1 gap across the model grid?

## Interactive App

The project includes **Performance Landscape Explorer**, a Streamlit + Plotly companion app for exploring the performance landscapes discussed in the report. It helps readers inspect where logistic regression performs well, where it collapses, and where generalization becomes less stable.

Run it from the project root:

```bash
streamlit run app/app.py
```

The app provides:

- Single-landscape heatmaps and 3D surfaces.
- Comparison mode for two selected landscapes.
- Preset views for the main findings.
- Interpretation panels for metrics, axes, and model behavior.
- Toggles for local/common color scales, 3D smoothing, and original grid-point overlays.

Smoothed 3D surfaces are visual interpolations between observed grid points.

## Run The App

Clone the repository and install dependencies:

```bash
git clone https://github.com/shacostapabley/performance-landscapes-text-classification.git
cd performance-landscapes-text-classification
pip install -r requirements.txt
streamlit run app/app.py
```
Alternatively, the app can be viewed at performancelandscapes.streamlit.app.

## Repository Structure

```text
.
|-- app/                         # Streamlit + Plotly app
|-- data/
|   |-- raw/                     # Local raw data, ignored by Git
|   |-- processed/               # Saved grid-search results
|   `-- README.md
|-- docs/
|   `-- project_summary.md
|-- figures/
|   |-- eda/
|   |-- heatmaps/
|   `-- surfaces/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_model_grid_experiment.ipynb
|   |-- 03_heatmap_generation.ipynb
|   |-- 04_surface_plot_exploration.ipynb
|   `-- archive/full_project_original.ipynb
|-- presentation/
|   `-- final_presentation.pdf
|-- report/
|   `-- final_report.pdf
|-- scripts/
|-- src/
|   `-- performance_landscapes/
|-- requirements.txt
`-- pyproject.toml
```

## Data

Raw Kaggle CSV files are intentionally ignored by Git and must be downloaded separately. Place them at:

```text
data/raw/Fake.csv
data/raw/True.csv
```

The app and visualization notebooks load the processed results table:

```text
data/processed/landscape_results.csv
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Notebook Workflow

1. `notebooks/01_eda.ipynb`: data loading, title cleaning, class balance checks, title length summaries, and token/bigram exploration.
2. `notebooks/02_model_grid_experiment.ipynb`: logistic-regression grid experiment.
3. `notebooks/03_heatmap_generation.ipynb`: static heatmap generation.
4. `notebooks/04_surface_plot_exploration.ipynb`: optional 3D surface exploration.

The archived full notebook is preserved at `notebooks/archive/full_project_original.ipynb`.

The modeling experiment is computationally expensive and should not be rerun unless you intentionally want to regenerate `data/processed/landscape_results.csv`.

## Key Findings

- L1 regularization creates a sharp collapse region at very small `C`.
- L2 regularization changes more smoothly across the grid.
- Unigram and unigram + bigram feature spaces often produce similar strong regions.
- Bigram-only models are weaker and less stable.
- The most useful region is a moderate-`C` performance plateau, not simply the weakest regularization extreme.

## Report And Presentation

- Report: `report/final_report.pdf`
- Presentation: `presentation/final_presentation.pdf`
