# Performance Landscape Explorer

This folder contains the Streamlit + Plotly app for exploring the model-performance landscapes from the project.

The app is a companion to the written report. It helps readers examine how feature representation, regularization, and feature budget shape logistic regression behavior across the full parameter grid. It loads:

```text
data/processed/landscape_results.csv
```

## Controls

The Single Landscape Explorer provides dropdown controls for:

- Representation: BoW or TF-IDF
- N-gram setting: Unigram, Uni+Bigram, or Bigram Only
- Penalty: L1 or L2
- Metric: Test F1, Test AUC, or F1 Gap

Each selection renders:

- A Plotly 2D heatmap
- A Plotly 3D surface
- Hover labels with Feature, Penalty, C, log10(C), max_features, and metric value

The app uses the Viridis color scale for readable interactive comparisons. Heatmaps are rendered on the 50-by-50 experiment grid so cells stay square rather than stretching with raw axis magnitudes.

## Finding Presets

The app includes four preset views:

1. L1 collapse
2. Unigram vs Bigram-only
3. BoW F1 valley vs AUC
4. Stability plateau

Each preset includes interpretation text tied to the project findings.

## Comparison Mode

Comparison Mode lets users select two landscapes side by side. It shows paired heatmaps and can optionally render paired 3D surfaces.

## Data Schema

The results CSV must include at least:

```text
Feature
Penalty
max_features
C
Test F1
Test AUC
F1 Gap
```

Extra columns are allowed and ignored unless useful.

## Run

From the project root:

```bash
streamlit run app/app.py
```

## Smoothing Note

Smoothed 3D surfaces are visual interpolations between observed grid points.

The initial 3D camera angle matches the original surface-plot exploration setting (`elev=40`, `azim=-140`). Plotly's reset-camera control returns to that default view.
