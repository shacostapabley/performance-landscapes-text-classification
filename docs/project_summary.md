# Project Summary

## Motivation

This project studies logistic regression behavior across a broad hyperparameter grid for title-based fake-news classification. Instead of reporting only one best model, it treats model performance as a landscape shaped by feature representation, n-gram choice, regularization penalty, regularization strength, and feature budget.

## Data Setup

The task uses article titles from a fake-versus-real news dataset. Raw CSV files should be stored locally in:

```text
data/raw/Fake.csv
data/raw/True.csv
```

Raw data are intentionally excluded from version control. The processed grid-search results used by the analysis and future app are stored in:

```text
data/processed/landscape_results.csv
```

## Modeling Setup

The experiment compares logistic regression models over:

- Bag-of-Words and TF-IDF representations.
- Unigram, unigram + bigram, and bigram-only feature spaces.
- L1 and L2 regularization.
- A grid over `C` and `max_features`.

The core metrics are Test F1, Test AUC, and F1 Gap.

## What The Heatmaps Represent

Each heatmap shows one metric over a two-dimensional grid:

- x-axis: regularization strength `C`
- y-axis: `max_features`
- color: selected metric value

Each grid point corresponds to an actually trained model from the saved experiment results.

## Main Findings

The analysis found several recurring landscape patterns:

- L1 regularization can produce sharp low-`C` collapse regions.
- L2 regularization tends to change more smoothly across the grid.
- Unigram and unigram + bigram settings often behave similarly.
- Bigram-only models are generally weaker and less stable.
- A moderate-`C` plateau provides strong performance without simply pushing toward the weakest regularization extreme.

## Limitations

- The analysis uses title text only.
- Runtime results are machine-dependent.
- The 3D surface images are presentation-style visualizations.
- Any smoothed surface should be interpreted as visual interpolation, not as new experimental evidence.

## Interactive Deliverable

The repository includes a Streamlit + Plotly app called **Performance Landscape Explorer**. The app serves as an interactive companion to the report, letting users inspect the landscapes through heatmaps, 3D surfaces, comparison mode, preset findings, and interpretation notes.
