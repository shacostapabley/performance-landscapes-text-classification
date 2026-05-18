import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


C_VALUES = np.logspace(-6, 6, 50)
MAX_FEATURES_VALUES = list(range(100, 20000, 406))
PENALTIES = ["l1", "l2"]


FEATURE_CONFIGS = {
    "BoW Unigram": {
        "vectorizer_class": CountVectorizer,
        "kwargs": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 1),
            "min_df": 2,
            "max_df": 0.95,
        },
    },
    "BoW Uni+Bigram": {
        "vectorizer_class": CountVectorizer,
        "kwargs": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
        },
    },
    "BoW Bigram Only": {
        "vectorizer_class": CountVectorizer,
        "kwargs": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (2, 2),
            "min_df": 2,
            "max_df": 0.95,
        },
    },
    "TF-IDF Unigram": {
        "vectorizer_class": TfidfVectorizer,
        "kwargs": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 1),
            "min_df": 2,
            "max_df": 0.95,
        },
    },
    "TF-IDF Uni+Bigram": {
        "vectorizer_class": TfidfVectorizer,
        "kwargs": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
        },
    },
    "TF-IDF Bigram Only": {
        "vectorizer_class": TfidfVectorizer,
        "kwargs": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (2, 2),
            "min_df": 2,
            "max_df": 0.95,
        },
    },
}


def evaluate_model(model, X_train_vec, X_test_vec, y_train, y_test):
    start_time = time.time()
    model.fit(X_train_vec, y_train)
    runtime = time.time() - start_time

    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)

    y_train_prob = model.predict_proba(X_train_vec)[:, 1]
    y_test_prob = model.predict_proba(X_test_vec)[:, 1]

    return {
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Train Precision": precision_score(y_train, y_train_pred, zero_division=0),
        "Train Recall": recall_score(y_train, y_train_pred, zero_division=0),
        "Train F1": f1_score(y_train, y_train_pred, zero_division=0),
        "Train AUC": roc_auc_score(y_train, y_train_prob),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Test Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Test Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "Test F1": f1_score(y_test, y_test_pred, zero_division=0),
        "Test AUC": roc_auc_score(y_test, y_test_prob),
        "F1 Gap": f1_score(y_train, y_train_pred, zero_division=0)
        - f1_score(y_test, y_test_pred, zero_division=0),
        "AUC Gap": roc_auc_score(y_train, y_train_prob)
        - roc_auc_score(y_test, y_test_prob),
        "Runtime Seconds": runtime,
    }


def run_grid_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_configs=FEATURE_CONFIGS,
    max_features_values=MAX_FEATURES_VALUES,
    C_values=C_VALUES,
    penalties=PENALTIES,
):
    results = []

    for feature_name, config in feature_configs.items():
        print(f"\nRunning feature setup: {feature_name}")

        vectorizer_class = config["vectorizer_class"]
        base_kwargs = config["kwargs"]

        for max_feat in max_features_values:
            vectorizer = vectorizer_class(
                max_features=max_feat,
                **base_kwargs,
            )

            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            print(f"  max_features = {max_feat}, train matrix shape = {X_train_vec.shape}")
            for penalty in penalties:
                for C in C_values:
                    model = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        max_iter=2000,
                        solver="liblinear",
                        random_state=42,
                    )

                    metrics = evaluate_model(
                        model,
                        X_train_vec,
                        X_test_vec,
                        y_train,
                        y_test,
                    )

                    results.append(
                        {
                            "Feature": feature_name,
                            "Vectorizer": "BoW" if "BoW" in feature_name else "TF-IDF",
                            "Ngram": feature_name.replace("BoW ", "").replace(
                                "TF-IDF ",
                                "",
                            ),
                            "max_features": max_feat,
                            "Penalty": penalty,
                            "C": C,
                            **metrics,
                        }
                    )

    return pd.DataFrame(results)

