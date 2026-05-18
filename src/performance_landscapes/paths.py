from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
HEATMAP_DIR = FIGURES_DIR / "heatmaps"
SURFACE_DIR = FIGURES_DIR / "surfaces"


def resolve_results_path():
    """Return the preferred saved grid-results path."""
    root_results = PROJECT_ROOT / "fake_news_title_only_logistic_results.csv"
    if root_results.exists():
        return root_results
    return PROCESSED_DATA_DIR / "landscape_results.csv"
