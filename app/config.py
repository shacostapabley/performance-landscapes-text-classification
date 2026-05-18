from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "landscape_results.csv"
REPORT_PATH = PROJECT_ROOT / "report" / "final_report.pdf"

APP_TITLE = "Performance Landscape Explorer"
APP_SUBTITLE = (
    "Interactive visualization of how feature representation, regularization, "
    "and metric choice shape logistic regression in text classification."
)

REPRESENTATIONS = ["BoW", "TF-IDF"]
NGRAM_SETTINGS = ["Unigram", "Uni+Bigram", "Bigram Only"]
PENALTIES = ["L1", "L2"]
METRICS = ["Test F1", "Test AUC", "F1 Gap"]

COLOR_SCALE = "Viridis"

DEFAULT_CAMERA_ELEV = 40
DEFAULT_CAMERA_AZIM = -140

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
}

PRESETS = {
    "L1 collapse": {
        "mode": "single",
        "left": {
            "representation": "BoW",
            "ngram": "Unigram",
            "penalty": "L1",
            "metric": "Test F1",
        },
        "interpretation": (
            "L1 creates a sharp low-C failure region. In this short, sparse "
            "title setting, strong L1 regularization can remove too many useful "
            "lexical features, causing the model to collapse toward near-constant "
            "predictions."
        ),
    },
    "Unigram vs Bigram-only": {
        "mode": "comparison",
        "left": {
            "representation": "TF-IDF",
            "ngram": "Unigram",
            "penalty": "L2",
            "metric": "Test F1",
        },
        "right": {
            "representation": "TF-IDF",
            "ngram": "Bigram Only",
            "penalty": "L2",
            "metric": "Test F1",
        },
        "interpretation": (
            "Unigram-inclusive models are stronger and more stable. Bigram-only "
            "models are weaker and more sensitive to feature budget, suggesting "
            "that phrase-level information alone is too sparse for short titles."
        ),
    },
    "BoW F1 valley vs AUC": {
        "mode": "comparison",
        "left": {
            "representation": "BoW",
            "ngram": "Uni+Bigram",
            "penalty": "L2",
            "metric": "Test F1",
        },
        "right": {
            "representation": "BoW",
            "ngram": "Uni+Bigram",
            "penalty": "L2",
            "metric": "Test AUC",
        },
        "interpretation": (
            "The BoW L2 valley appears in F1 but disappears in AUC. This "
            "suggests the region is more related to final classification behavior "
            "than to a full loss of overall ranking ability."
        ),
    },
    "Stability plateau": {
        "mode": "comparison",
        "left": {
            "representation": "TF-IDF",
            "ngram": "Unigram",
            "penalty": "L2",
            "metric": "Test F1",
        },
        "right": {
            "representation": "TF-IDF",
            "ngram": "Unigram",
            "penalty": "L2",
            "metric": "F1 Gap",
        },
        "interpretation": (
            "High performance is not always the best region. In many settings, "
            "F1 reaches a broad plateau before the train-test gap becomes largest, "
            "so moderate-C regions can offer a better performance-stability tradeoff."
        ),
    },
}
