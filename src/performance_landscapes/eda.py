import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


EXTRA_BAD_TOKENS = {
    "s",
    "t",
    "u",
    "re",
    "ve",
    "ll",
    "d",
    "m",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS).union(EXTRA_BAD_TOKENS)


def clean_tokenize(text):
    text = text.lower()
    text = re.sub(r"[']|[’]", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOP_WORDS]


simple_tokenize = clean_tokenize


def get_bigrams(tokens):
    return list(zip(tokens, tokens[1:]))


def count_tokens_by_class(df, tokenizer=clean_tokenize):
    real_tokens = []
    fake_tokens = []

    for title in df[df["label"] == 0]["title"]:
        real_tokens.extend(tokenizer(title))

    for title in df[df["label"] == 1]["title"]:
        fake_tokens.extend(tokenizer(title))

    return Counter(real_tokens), Counter(fake_tokens)


def count_bigrams_by_class(df, tokenizer=clean_tokenize):
    real_bigrams = []
    fake_bigrams = []

    for title in df[df["label"] == 0]["title"]:
        real_bigrams.extend(get_bigrams(tokenizer(title)))

    for title in df[df["label"] == 1]["title"]:
        fake_bigrams.extend(get_bigrams(tokenizer(title)))

    return Counter(real_bigrams), Counter(fake_bigrams)


def plot_top_words(counter, title, n=15):
    top_words = counter.most_common(n)
    words = [w for w, c in top_words]
    counts = [c for w, c in top_words]

    cmap = plt.get_cmap("inferno")
    plt.figure(figsize=(8, 5))
    plt.barh(
        words[::-1],
        counts[::-1],
        color=tuple(cmap(x) for x in np.linspace(0.7, 0.1, n)),
    )
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()


def plot_top_bigrams(counter, title, n=10):
    top_bigrams = counter.most_common(n)
    labels = [" ".join(bg) for bg, c in top_bigrams]
    counts = [c for bg, c in top_bigrams]

    plt.figure(figsize=(9, 5))
    plt.barh(labels[::-1], counts[::-1])
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()

