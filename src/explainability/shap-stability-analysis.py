#!/usr/bin/env python3

"""
SHAP Explanation Stability Across Cross-Validation Folds

Purpose
-------
Assess whether SHAP feature importance remains stable across
cross-validation folds for a Random Forest classifier.

Outputs
-------
results/
    shap_fold_importance.csv
    shap_stability_summary.csv
    shap_fold_ranks.csv
    shap_rank_correlation.csv
    shap_importance_mean.png
    shap_importance_stability.png
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, spearmanr

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

import shap


warnings.filterwarnings("ignore")


# Configuration
# ===========================================================

CONFIG_FILE = "config/config.yaml"

MODEL_FILE = "results/models/best_model.pkl"

OUTPUT_DIR = Path("results/shap-cross-val")

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42


# ============================================================
# Utility functions
# ============================================================

def create_output_directory(path: str) -> Path:
    """Create output directory when missing."""
    output_dir = Path(path)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to create output directory: {output_dir}"
        ) from exc

    return output_dir


# ===========================================================
# Load Dataset
# ===========================================================

def load_data():
    """
    Load the dataset and construct all engineered features.

    Returns
    -------
    X : DataFrame

    y : Series
    """

    config = load_config(CONFIG_FILE)

    df = load_dataset(
        config["data"]["raw_file"]
    )
    df = df.drop(columns=["Gestational age at delivery [weeks]"])
    df = df.drop(columns=["Birth weight [g]"])

    df = build_interaction_features(
        df,
        config
    )

    target = config["target"]["column"]

    features = (
        config["features"]["clinical"]
        + config["features"]["doppler"]
        + config["features"]["biomarkers"]
        + config["features"]["interaction_features"]
    )

    features = [
        f for f in features
        if f in df.columns
    ]

    X = df[features]

    y = df[target]

    return X, y


def ensure_binary_target(y: pd.Series) -> pd.Series:
    """Convert a binary target to numerical labels."""

    unique_values = list(pd.unique(y))

    if len(unique_values) != 2:
        raise ValueError(
            "This SHAP stability program expects a binary target.\n"
            f"Detected classes: {unique_values}"
        )

    mapping = {
        unique_values[0]: 0,
        unique_values[1]: 1
    }

    return y.map(mapping).astype(int)


def calculate_shap_importance(
    model,
    X_validation: pd.DataFrame
) -> np.ndarray:
    """
    Calculate mean absolute SHAP value for each feature.

    For binary Random Forest classification, SHAP can return
    either a list or an array depending on the SHAP version.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_validation)

    # Older SHAP versions:
    # shap_values = [class_0_values, class_1_values]
    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            values = shap_values[1]
        else:
            values = shap_values[0]

    else:
        values = np.asarray(shap_values)

        # Newer SHAP versions may return:
        # samples x features x classes
        if values.ndim == 3:

            if values.shape[-1] == 2:
                values = values[:, :, 1]

            elif values.shape[0] == 2:
                values = values[1]

            else:
                values = values[:, :, 0]

        elif values.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP shape: {values.shape}"
            )

    mean_abs_shap = np.mean(
        np.abs(values),
        axis=0
    )

    return mean_abs_shap


def rank_features(
    importance: pd.Series
) -> pd.Series:
    """
    Convert feature importance values into descending ranks.
    Rank 1 = most important feature.
    """

    return importance.rank(
        ascending=False,
        method="average"
    )


# ============================================================
# Stability calculations
# ============================================================

def calculate_rank_correlation(
    rank_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate Spearman rank correlation between every
    pair of cross-validation folds.
    """

    folds = rank_df.columns
    rows = []

    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):

            fold_a = folds[i]
            fold_b = folds[j]

            rho, p_value = spearmanr(
                rank_df[fold_a],
                rank_df[fold_b]
            )

            rows.append({
                "fold_1": fold_a,
                "fold_2": fold_b,
                "spearman_rho": rho,
                "p_value": p_value
            })

    return pd.DataFrame(rows)

# ============================================================
# Main analysis
# ============================================================

def run_stability_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    n_splits: int = 5,
    n_estimators: int = 500
) -> None:

    print("\nStarting SHAP stability analysis")
    print("-" * 60)
    print(f"Samples:       {len(X)}")
    print(f"Features:      {X.shape[1]}")
    print(f"CV folds:      {n_splits}")
    print(f"Trees/fold:    {n_estimators}")
    print("-" * 60)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_importance = []
    fold_rankings = []

    fold_auc = []
    fold_f1 = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(f"\nFold {fold_number}/{n_splits}")

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE + fold_number,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt"
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Prediction performance
        # ----------------------------------------------------

        probability = model.predict_proba(X_valid)[:, 1]

        prediction = (
            probability >= 0.50
        ).astype(int)

        auc = roc_auc_score(
            y_valid,
            probability
        )

        f1 = f1_score(
            y_valid,
            prediction,
            zero_division=0
        )

        fold_auc.append(auc)
        fold_f1.append(f1)

        print(
            f"  ROC-AUC = {auc:.4f}"
            f" | F1 = {f1:.4f}"
        )

        # ----------------------------------------------------
        # SHAP values
        # ----------------------------------------------------

        shap_importance = calculate_shap_importance(
            model,
            X_valid
        )

        importance_series = pd.Series(
            shap_importance,
            index=X.columns,
            name=f"fold_{fold_number}"
        )

        ranking_series = rank_features(
            importance_series
        )

        fold_importance.append(
            importance_series
        )

        fold_rankings.append(
            ranking_series
        )

        print(
            "  Top features:",
            list(
                importance_series
                .sort_values(ascending=False)
                .head(5)
                .index
            )
        )

    # ========================================================
    # Convert fold results to tables
    # ========================================================

    importance_df = pd.concat(
        fold_importance,
        axis=1
    )

    importance_df.columns = [
        f"fold_{i}"
        for i in range(1, n_splits + 1)
    ]

    rank_df = pd.concat(
        fold_rankings,
        axis=1
    )

    rank_df.columns = [
        f"fold_{i}"
        for i in range(1, n_splits + 1)
    ]

    # ========================================================
    # Stability summary
    # ========================================================

    stability_df = pd.DataFrame(index=X.columns)

    stability_df["mean_shap"] = (
        importance_df.mean(axis=1)
    )

    stability_df["std_shap"] = (
        importance_df.std(axis=1)
    )

    stability_df["cv_shap"] = (
        stability_df["std_shap"]
        /
        stability_df["mean_shap"].replace(0, np.nan)
    )

    stability_df["mean_rank"] = (
        rank_df.mean(axis=1)
    )

    stability_df["std_rank"] = (
        rank_df.std(axis=1)
    )

    stability_df["min_rank"] = (
        rank_df.min(axis=1)
    )

    stability_df["max_rank"] = (
        rank_df.max(axis=1)
    )

    stability_df["rank_range"] = (
        stability_df["max_rank"]
        -
        stability_df["min_rank"]
    )

    # Frequency among top-10 features
    top_k = min(10, len(X.columns))

    stability_df["top10_frequency"] = (
        rank_df <= top_k
    ).mean(axis=1)

    # Frequency among top-5 features
    top_k5 = min(5, len(X.columns))

    stability_df["top5_frequency"] = (
        rank_df <= top_k5
    ).mean(axis=1)

    stability_df = stability_df.sort_values(
        "mean_shap",
        ascending=False
    )

    # ========================================================
    # Rank stability
    # ========================================================


    correlation_df = calculate_rank_correlation(
        rank_df
    )

    # ========================================================
    # Save numerical results
    # ========================================================

    importance_path = (
        output_dir /
        "shap_fold_importance.csv"
    )

    stability_path = (
        output_dir /
        "shap_stability_summary.csv"
    )

    ranks_path = (
        output_dir /
        "shap_fold_ranks.csv"
    )

    correlation_path = (
        output_dir /
        "shap_rank_correlation.csv"
    )

    importance_df.to_csv(
        importance_path
    )

    stability_df.to_csv(
        stability_path
    )

    rank_df.to_csv(
        ranks_path
    )

    correlation_df.to_csv(
        correlation_path,
        index=False
    )

    # ========================================================
    # Create summary report
    # ========================================================

    report_path = (
        output_dir /
        "shap_stability_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "SHAP Explanation Stability Analysis\n"
        )
        report.write(
            "=" * 50 + "\n\n"
        )

        report.write(
            f"Number of samples: {len(X)}\n"
        )

        report.write(
            f"Number of features: {X.shape[1]}\n"
        )

        report.write(
            f"Cross-validation folds: {n_splits}\n"
        )

        report.write(
            f"Random Forest estimators: {n_estimators}\n\n"
        )

        report.write(
            "Predictive performance\n"
        )
        report.write(
            "-" * 30 + "\n"
        )

        report.write(
            f"Mean ROC-AUC: "
            f"{np.mean(fold_auc):.4f}\n"
        )

        report.write(
            f"STD ROC-AUC: "
            f"{np.std(fold_auc, ddof=1):.4f}\n"
        )

        report.write(
            f"Mean F1: "
            f"{np.mean(fold_f1):.4f}\n"
        )

        report.write(
            f"STD F1: "
            f"{np.std(fold_f1, ddof=1):.4f}\n\n"
        )

        report.write(
            "Explanation stability\n"
        )
        report.write(
            "-" * 30 + "\n"
        )

        report.write(
            "Top 20 features by mean absolute SHAP\n"
        )
        report.write(
            "-" * 30 + "\n"
        )

        top_features = stability_df.head(20)

        for feature, row in top_features.iterrows():

            report.write(
                f"{feature}: "
                f"mean SHAP={row['mean_shap']:.6f}, "
                f"std={row['std_shap']:.6f}, "
                f"mean rank={row['mean_rank']:.2f}, "
                f"top-10 frequency="
                f"{row['top10_frequency']:.2f}\n"
            )

    # ========================================================
    # Plot 1: mean SHAP importance
    # ========================================================

    plot_features = stability_df.head(15)

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_features.index[::-1],
        plot_features["mean_shap"][::-1]
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Mean SHAP Feature Importance Across Cross-Validation Folds"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "shap_importance_mean.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Plot 2: mean SHAP ± standard deviation
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )
    
    plt.barh(
        plot_features.index[::-1],
        plot_features["mean_shap"][::-1],
        xerr=plot_features["std_shap"][::-1],
        capsize=3,
        color='red'
    )

    plt.xlabel(
        "Mean absolute SHAP value ± SD"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SHAP Explanation Stability Across Cross-Validation Folds"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "shap_importance_stability.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Final console output
    # ========================================================

    print("\n" + "=" * 60)
    print("SHAP STABILITY RESULTS")
    print("=" * 60)

    print(
        f"Mean ROC-AUC: "
        f"{np.mean(fold_auc):.4f} "
        f"+/- {np.std(fold_auc, ddof=1):.4f}"
    )

    print(
        f"Mean F1:      "
        f"{np.mean(fold_f1):.4f} "
        f"+/- {np.std(fold_f1, ddof=1):.4f}"
    )

    print("\nTop 10 stable features:")

    display_columns = [
        "mean_shap",
        "std_shap",
        "mean_rank",
        "rank_range",
        "top10_frequency"
    ]

    print(
        stability_df.head(10)[display_columns]
    )

    print("\nFiles saved in:")
    print(output_dir.resolve())


# ============================================================
# Command-line interface
# ============================================================

def main() -> None:

    X, y = load_data()
    
    y = y.isin(
        ["PE", "IUGR+PE"]
    ).astype(int)   
    
    print(y)
    
    output_dir = create_output_directory(
        OUTPUT_DIR
    )
    
    
    run_stability_analysis(
        X=X,
        y=y,
        output_dir=output_dir,
        n_splits=5,
        n_estimators=500
    )
    

if __name__ == "__main__":
    main()