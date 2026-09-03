"""End-to-end customer churn prediction pipeline.

Loads the IBM Telco Customer Churn dataset, runs EDA with ``eda-kit``,
preprocesses the data, trains a Random Forest classifier (plus a Dummy
baseline), evaluates it and saves the metrics and plots to ``results/``.

Run:  python src/train_model.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import eda_kit as ek

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_PATH = ROOT / "data" / "telco_churn.csv"
RESULTS = ROOT / "results"
TARGET = "Churn"

sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    """Download (if needed) and load the dataset."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type fixes and missing-value handling."""
    df = df.copy()
    # TotalCharges has a few blank strings -> coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # drop the handful of rows with missing TotalCharges (they are new customers)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals and cap outliers using eda-kit."""
    df = df.copy()
    cat_cols, num_cols, card_cols = ek.grab_col_names(df, cat_th=10, car_th=20)

    # drop high-cardinality / non-predictive columns
    df = df.drop(columns=card_cols)

    # cap outliers instead of dropping rows
    for col in num_cols:
        if ek.check_outlier(df, col):
            ek.replace_with_thresholds(df, col)

    # collapse rare classes, then one-hot encode (keep target out)
    encode_cols = [c for c in cat_cols if c != TARGET]
    df = ek.rare_encoder(df, rare_perc=0.01)
    df = ek.one_hot_encoder(df, encode_cols)
    return df


def evaluate(model, X_te, y_te) -> dict:
    """Return a dict of classification metrics."""
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba),
    }


def plot_confusion(model, X_te, y_te, path: Path) -> None:
    cm = confusion_matrix(y_te, model.predict(X_te))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOAD DATA")
    print("=" * 60)
    df = load_data()
    print("raw shape:", df.shape)

    print("=" * 60)
    print("EDA (eda-kit)")
    print("=" * 60)
    ek.check_df(df)

    print("=" * 60)
    print("CLEAN + PREPROCESS")
    print("=" * 60)
    df = clean(df)
    df = preprocess(df)
    print("final shape:", df.shape)

    print("=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == "Yes").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("train:", X_tr.shape, "| test:", X_te.shape)

    print("=" * 60)
    print("BASELINE (DummyClassifier)")
    print("=" * 60)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base_metrics = evaluate(baseline, X_te, y_te)
    print(base_metrics)

    print("=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr)
    metrics = evaluate(model, X_te, y_te)
    print(metrics)

    # persist metrics
    metrics_df = pd.DataFrame([base_metrics, metrics], index=["Baseline", "RandomForest"])
    metrics_df.to_csv(RESULTS / "metrics.csv")
    print("\nmetrics saved ->", RESULTS / "metrics.csv")

    # plots
    plot_confusion(model, X_te, y_te, RESULTS / "confusion_matrix.png")
    ek.plot_importance(model, X_tr, num=12, save=str(RESULTS / "feature_importance.png"))
    print("plots saved ->", RESULTS)

    # top drivers
    imp = pd.DataFrame(
        {"feature": X_tr.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTOP 10 CHURN DRIVERS:")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
