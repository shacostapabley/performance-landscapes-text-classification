import pandas as pd
from sklearn.model_selection import train_test_split

from performance_landscapes.paths import RAW_DATA_DIR


def load_raw_news_data(data_dir=RAW_DATA_DIR):
    """Load and combine the raw fake/true news CSV files."""
    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = pd.read_csv(data_dir / "True.csv")

    fake_df["label"] = 1
    true_df["label"] = 0

    return pd.concat([fake_df, true_df], ignore_index=True)


def clean_title_rows(df):
    """Keep rows with non-empty title text."""
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    return df[df["title"].str.len() > 0].copy()


def balanced_subset(df, subset_size=10000, random_state=42):
    """Take the same balanced optional subset used in the archived notebook."""
    if subset_size is None or subset_size >= len(df):
        return df

    fake_sample = df[df["label"] == 1].sample(
        n=subset_size // 2,
        random_state=random_state,
    )
    real_sample = df[df["label"] == 0].sample(
        n=subset_size - (subset_size // 2),
        random_state=random_state,
    )
    return pd.concat([fake_sample, real_sample], ignore_index=True)


def split_title_data(df, test_size=0.2, random_state=42):
    """Create the title-only train/test split used for the model grid."""
    X = df["title"]
    y = df["label"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
