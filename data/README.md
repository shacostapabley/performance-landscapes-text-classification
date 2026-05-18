# Data

The raw dataset files are not tracked in Git because they are large and should not be committed:

```text
data/raw/Fake.csv
data/raw/True.csv
```

The original dataset can be downloaded from Kaggle at https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets. This project uses fake-and-real news dataset with separate `Fake.csv` and `True.csv` files.

The notebooks expect those raw files locally for EDA and model-grid reruns. The saved experiment results used by the visualization notebooks and future app are stored in:

```text
data/processed/landscape_results.csv
```

`landscape_results.csv` contains one row per trained grid point from the logistic-regression landscape experiment.

Expected key columns for the future app:

```text
Feature
Penalty
max_features
C
Test F1
Test AUC
F1 Gap
```
