#!/usr/bin/env python3

"""
Local SHAP Explanation Stability Across Repeated Cross-Validation

Purpose
-------
Assess whether patient-level SHAP explanations remain stable when
the same patient is evaluated by independently trained models.

Recommended design:
    5-fold Stratified CV × 10 repeats

Each patient is therefore in a validation fold once per repeat,
giving 10 out-of-sample local explanations per patient.

Outputs
-------
results/local_shap_stability/
    local_shap_values.csv
    local_shap_summary.csv
    local_top_features.csv
    local_prediction_stability.csv
    local_patient_stability.csv
    local_jaccard.csv
    local_rank_correlation.csv
    local_shap_stability_report.txt
    local_explanation_stability.png
    local_probability_stability.png
"""

from __future__ import annotations

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

CONFIG_FILE = "config/config.yaml"

MODEL_FILE = "results/models/best_model.pkl"

OUTPUT_DIR = Path("results/local-shap-stability")

RANDOM_STATE = 42

DEFAULT_FOLDS = 5
DEFAULT_REPEATS = 10
DEFAULT_TREES = 500

TOP_K = 5


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


# ============================================================
# SHAP extraction
# ============================================================

def get_positive_class_shap(
    model,
    X_valid: pd.DataFrame
) -> np.ndarray:

    """
    Returns:
        samples x features

    SHAP values for the positive class.
    """

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_valid
    )

    # Older SHAP
    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) != 2:
            raise ValueError(
                "Unexpected SHAP output."
            )

        values = np.asarray(
            shap_values[1]
        )

    else:

        values = np.asarray(
            shap_values
        )

        # Newer SHAP:
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

    return values


# ============================================================
# Local ranking
# ============================================================

def rank_shap_features(
    shap_vector: np.ndarray,
    feature_names: list[str]
) -> pd.Series:

    values = pd.Series(
        np.abs(shap_vector),
        index=feature_names
    )

    return values.rank(
        ascending=False,
        method="average"
    )


def get_top_features(
    shap_vector: np.ndarray,
    feature_names: list[str],
    k: int = TOP_K
) -> list[str]:

    values = pd.Series(
        np.abs(shap_vector),
        index=feature_names
    )

    return list(
        values
        .sort_values(
            ascending=False
        )
        .head(k)
        .index
    )


# ============================================================
# Jaccard similarity
# ============================================================

def jaccard_similarity(
    set_a: set[str],
    set_b: set[str]
) -> float:

    union = set_a | set_b

    if len(union) == 0:
        return np.nan

    return (
        len(set_a & set_b)
        /
        len(union)
    )


# ============================================================
# Main analysis
# ============================================================

def run_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    folds: int = DEFAULT_FOLDS,
    repeats: int = DEFAULT_REPEATS,
    trees: int = DEFAULT_TREES
) -> None:

    feature_names = list(
        X.columns
    )

    n_samples = len(X)

    print("\n" + "=" * 70)
    print("LOCAL SHAP EXPLANATION STABILITY")
    print("=" * 70)

    print(
        f"Samples:       {n_samples}"
    )

    print(
        f"Features:      {len(feature_names)}"
    )

    print(
        f"Folds:         {folds}"
    )

    print(
        f"Repeats:       {repeats}"
    )

    print(
        f"RF trees:      {trees}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Repeated stratified CV
    # --------------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    shap_records = []

    prediction_records = []

    performance_auc = []
    performance_f1 = []

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    for split_id, (
        train_idx,
        valid_idx
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        repeat_id = (
            (split_id - 1)
            // folds
        ) + 1

        fold_id = (
            (split_id - 1)
            % folds
        ) + 1

        split_name = (
            f"R{repeat_id:02d}_F{fold_id:02d}"
        )

        print(
            f"\n{split_name}"
        )

        X_train = X.iloc[
            train_idx
        ].copy()

        X_valid = X.iloc[
            valid_idx
        ].copy()

        y_train = y.iloc[
            train_idx
        ]

        y_valid = y.iloc[
            valid_idx
        ]

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=trees,
            random_state=(
                RANDOM_STATE
                + split_id
            ),
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        auc = roc_auc_score(
            y_valid,
            probabilities
        )

        f1 = f1_score(
            y_valid,
            predictions,
            zero_division=0
        )

        performance_auc.append(
            auc
        )

        performance_f1.append(
            f1
        )

        print(
            f"ROC-AUC = {auc:.4f} | "
            f"F1 = {f1:.4f}"
        )

        # ----------------------------------------------------
        # Local SHAP values
        # ----------------------------------------------------

        shap_values = (
            get_positive_class_shap(
                model,
                X_valid
            )
        )

        # ----------------------------------------------------
        # Store patient-level results
        # ----------------------------------------------------

        for local_position, sample_idx in enumerate(
            valid_idx
        ):

            shap_vector = (
                shap_values[
                    local_position
                ]
            )

            patient_id = int(
                X.index[sample_idx]
            )

            top_features = (
                get_top_features(
                    shap_vector,
                    feature_names,
                    TOP_K
                )
            )

            ranks = (
                rank_shap_features(
                    shap_vector,
                    feature_names
                )
            )

            # ------------------------------------------------
            # SHAP values
            # ------------------------------------------------

            for feature, value in zip(
                feature_names,
                shap_vector
            ):

                shap_records.append({
                    "patient_id": patient_id,
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "split": split_name,
                    "feature": feature,
                    "shap_value": float(value),
                    "abs_shap": float(
                        abs(value)
                    ),
                    "rank": float(
                        ranks[feature]
                    ),
                    "is_top5": (
                        feature
                        in top_features
                    )
                })

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction_records.append({
                "patient_id": patient_id,
                "repeat": repeat_id,
                "fold": fold_id,
                "split": split_name,
                "true_class": int(
                    y_valid.loc[
                        X.index[sample_idx]
                    ]
                ),
                "probability": float(
                    probabilities[
                        local_position
                    ]
                ),
                "prediction": int(
                    predictions[
                        local_position
                    ]
                )
            })

    # ========================================================
    # SHAP long-format table
    # ========================================================

    shap_df = pd.DataFrame(
        shap_records
    )

    prediction_df = pd.DataFrame(
        prediction_records
    )

    # ========================================================
    # Patient-level SHAP stability
    # ========================================================

    summary_records = []

    patients = (
        shap_df[
            "patient_id"
        ]
        .unique()
    )

    for patient_id in patients:

        patient_data = shap_df[
            shap_df["patient_id"]
            == patient_id
        ]

        # ----------------------------------------------------
        # Explanation count
        # ----------------------------------------------------

        explanation_count = (
            patient_data["split"]
            .nunique()
        )

        # ----------------------------------------------------
        # Feature-level statistics
        # ----------------------------------------------------

        feature_stats = (
            patient_data
            .groupby("feature")
            ["abs_shap"]
            .agg(
                mean="mean",
                std="std"
            )
        )

        feature_stats["cv"] = (
            feature_stats["std"]
            /
            feature_stats["mean"].replace(
                0,
                np.nan
            )
        )

        feature_stats["mean_rank"] = (
            patient_data
            .groupby("feature")
            ["rank"]
            .mean()
        )

        feature_stats["rank_std"] = (
            patient_data
            .groupby("feature")
            ["rank"]
            .std()
        )

        feature_stats["top5_frequency"] = (
            patient_data
            .groupby("feature")
            ["is_top5"]
            .mean()
        )

        # ----------------------------------------------------
        # Top feature
        # ----------------------------------------------------

        top_feature = (
            feature_stats
            .sort_values(
                "mean",
                ascending=False
            )
            .index[0]
        )

        top_feature_frequency = (
            feature_stats
            .loc[
                top_feature,
                "top5_frequency"
            ]
        )

        # ----------------------------------------------------
        # Mean explanation rank
        # ----------------------------------------------------

        mean_rank = (
            feature_stats
            .sort_values(
                "mean",
                ascending=False
            )
            ["mean_rank"]
            .iloc[:TOP_K]
            .mean()
        )

        # ----------------------------------------------------
        # Pairwise top-5 Jaccard
        # ----------------------------------------------------

        top_sets = []

        for split_name, split_data in (
            patient_data
            .groupby("split")
        ):

            top_set = set(
                split_data
                .sort_values(
                    "abs_shap",
                    ascending=False
                )
                .head(TOP_K)
                ["feature"]
            )

            top_sets.append(
                top_set
            )

        jaccard_values = []

        for set_a, set_b in combinations(
            top_sets,
            2
        ):

            jaccard_values.append(
                jaccard_similarity(
                    set_a,
                    set_b
                )
            )

        mean_jaccard = (
            np.mean(
                jaccard_values
            )
            if jaccard_values
            else np.nan
        )

        # ----------------------------------------------------
        # SHAP sign stability
        # ----------------------------------------------------

        sign_data = (
            patient_data
            .copy()
        )

        sign_data["sign"] = np.sign(
            sign_data["shap_value"]
        )

        positive_fraction = (
            sign_data
            .groupby("feature")
            ["sign"]
            .apply(
                lambda x:
                np.mean(x > 0)
            )
        )

        negative_fraction = (
            sign_data
            .groupby("feature")
            ["sign"]
            .apply(
                lambda x:
                np.mean(x < 0)
            )
        )

        sign_consistency = pd.concat(
            [
                positive_fraction,
                negative_fraction
            ],
            axis=1
        ).max(
            axis=1
        )

        mean_sign_consistency = (
            sign_consistency
            .sort_values(
                ascending=False
            )
            .head(TOP_K)
            .mean()
        )

        # ----------------------------------------------------
        # Explanation magnitude stability
        # ----------------------------------------------------

        mean_abs_shap = (
            feature_stats[
                "mean"
            ]
            .sort_values(
                ascending=False
            )
            .head(TOP_K)
            .mean()
        )

        mean_cv = (
            feature_stats[
                "cv"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .sort_values(
                ascending=True
            )
            .head(TOP_K)
            .mean()
        )

        # ----------------------------------------------------
        # Local stability score
        #
        # Combines:
        #   top-feature recurrence
        #   top-k Jaccard
        #   sign consistency
        # ----------------------------------------------------

        local_stability_score = np.mean(
            [
                top_feature_frequency,
                mean_jaccard,
                mean_sign_consistency
            ]
        )

        summary_records.append({
            "patient_id": patient_id,
            "explanation_count": explanation_count,
            "top_feature": top_feature,
            "top_feature_frequency": (
                top_feature_frequency
            ),
            "mean_top5_rank": mean_rank,
            "mean_jaccard_top5": (
                mean_jaccard
            ),
            "mean_top5_sign_consistency": (
                mean_sign_consistency
            ),
            "mean_top5_abs_shap": (
                mean_abs_shap
            ),
            "mean_top5_cv": mean_cv,
            "local_stability_score": (
                local_stability_score
            )
        })

    patient_summary = pd.DataFrame(
        summary_records
    )

    # ========================================================
    # Prediction stability
    # ========================================================

    prediction_summary = (
        prediction_df
        .groupby("patient_id")
        .agg(
            explanation_count=(
                "probability",
                "count"
            ),
            mean_probability=(
                "probability",
                "mean"
            ),
            std_probability=(
                "probability",
                "std"
            ),
            min_probability=(
                "probability",
                "min"
            ),
            max_probability=(
                "probability",
                "max"
            ),
            mean_prediction=(
                "prediction",
                "mean"
            )
        )
        .reset_index()
    )

    prediction_summary[
        "prediction_stability"
    ] = np.where(
        prediction_summary[
            "mean_prediction"
        ] >= 0.5,
        prediction_summary[
            "mean_prediction"
        ],
        1.0 -
        prediction_summary[
            "mean_prediction"
        ]
    )

    # ========================================================
    # Feature-level local summary
    # ========================================================

    local_feature_summary = (
        shap_df
        .groupby(
            [
                "patient_id",
                "feature"
            ]
        )
        .agg(
            mean_abs_shap=(
                "abs_shap",
                "mean"
            ),
            std_abs_shap=(
                "abs_shap",
                "std"
            ),
            mean_rank=(
                "rank",
                "mean"
            ),
            std_rank=(
                "rank",
                "std"
            ),
            top5_frequency=(
                "is_top5",
                "mean"
            )
        )
        .reset_index()
    )

    # ========================================================
    # Save results
    # ========================================================

    shap_df.to_csv(
        output_dir /
        "local_shap_values.csv",
        index=False
    )

    local_feature_summary.to_csv(
        output_dir /
        "local_shap_summary.csv",
        index=False
    )

    patient_summary.to_csv(
        output_dir /
        "local_patient_stability.csv",
        index=False
    )

    prediction_df.to_csv(
        output_dir /
        "local_top_features.csv",
        index=False
    )

    prediction_summary.to_csv(
        output_dir /
        "local_prediction_stability.csv",
        index=False
    )

    # ========================================================
    # Aggregate feature importance
    # ========================================================

    global_local_summary = (
        shap_df
        .groupby("feature")
        .agg(
            mean_abs_shap=(
                "abs_shap",
                "mean"
            ),
            std_abs_shap=(
                "abs_shap",
                "std"
            ),
            mean_rank=(
                "rank",
                "mean"
            ),
            top5_frequency=(
                "is_top5",
                "mean"
            )
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
    )

    global_local_summary.to_csv(
        output_dir /
        "local_explanation_global_summary.csv"
    )

    # ========================================================
    # Patient stability plot
    # ========================================================

    plot_data = (
        patient_summary
        .sort_values(
            "local_stability_score"
        )
        .tail(20)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_data["patient_id"].astype(str),
        plot_data[
            "local_stability_score"
        ]
    )

    plt.xlabel(
        "Local explanation stability score"
    )

    plt.ylabel(
        "Patient"
    )

    plt.title(
        "Patient-Level SHAP Explanation Stability"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "local_explanation_stability.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Probability stability plot
    # ========================================================

    probability_plot = (
        prediction_summary
        .sort_values(
            "std_probability"
        )
        .head(20)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.errorbar(
        probability_plot[
            "patient_id"
        ].astype(str),
        probability_plot[
            "mean_probability"
        ],
        yerr=probability_plot[
            "std_probability"
        ],
        fmt="o"
    )

    plt.xlabel(
        "Patient"
    )

    plt.ylabel(
        "Predicted probability ± SD"
    )

    plt.title(
        "Patient-Level Prediction Stability"
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "local_probability_stability.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Report
    # ========================================================

    report_path = (
        output_dir /
        "local_shap_stability_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "Local SHAP Explanation Stability\n"
        )

        report.write(
            "=" * 60 + "\n\n"
        )

        report.write(
            f"Samples: {n_samples}\n"
        )

        report.write(
            f"Features: {len(feature_names)}\n"
        )

        report.write(
            f"Folds: {folds}\n"
        )

        report.write(
            f"Repeats: {repeats}\n"
        )

        report.write(
            f"Trees: {trees}\n\n"
        )

        report.write(
            "Predictive performance\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            f"Mean ROC-AUC: "
            f"{np.mean(performance_auc):.4f}\n"
        )

        report.write(
            f"SD ROC-AUC: "
            f"{np.std(performance_auc, ddof=1):.4f}\n"
        )

        report.write(
            f"Mean F1: "
            f"{np.mean(performance_f1):.4f}\n"
        )

        report.write(
            f"SD F1: "
            f"{np.std(performance_f1, ddof=1):.4f}\n\n"
        )

        report.write(
            "Patient-level explanation stability\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            f"Mean local stability score: "
            f"{patient_summary['local_stability_score'].mean():.4f}\n"
        )

        report.write(
            f"Median local stability score: "
            f"{patient_summary['local_stability_score'].median():.4f}\n"
        )

        report.write(
            f"Mean top-feature frequency: "
            f"{patient_summary['top_feature_frequency'].mean():.4f}\n"
        )

        report.write(
            f"Mean top-5 Jaccard: "
            f"{patient_summary['mean_jaccard_top5'].mean():.4f}\n"
        )

        report.write(
            f"Mean top-5 sign consistency: "
            f"{patient_summary['mean_top5_sign_consistency'].mean():.4f}\n\n"
        )

        report.write(
            "Top recurrent local features\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        for feature, row in (
            global_local_summary
            .head(15)
            .iterrows()
        ):

            report.write(
                f"{feature}: "
                f"mean abs SHAP="
                f"{row['mean_abs_shap']:.6f}, "
                f"mean rank="
                f"{row['mean_rank']:.2f}, "
                f"top-5 frequency="
                f"{row['top5_frequency']:.3f}\n"
            )

    # ========================================================
    # Console
    # ========================================================

    print("\n" + "=" * 70)
    print("LOCAL EXPLANATION STABILITY RESULTS")
    print("=" * 70)

    print(
        f"Mean local stability score: "
        f"{patient_summary['local_stability_score'].mean():.4f}"
    )

    print(
        f"Mean top-feature frequency: "
        f"{patient_summary['top_feature_frequency'].mean():.4f}"
    )

    print(
        f"Mean top-5 Jaccard: "
        f"{patient_summary['mean_jaccard_top5'].mean():.4f}"
    )

    print(
        f"Mean top-5 sign consistency: "
        f"{patient_summary['mean_top5_sign_consistency'].mean():.4f}"
    )

    print("\nMost recurrent patient-level top features:")

    print(
        global_local_summary.head(10)
    )

    print("\nResults saved to:")
    print(
        output_dir.resolve()
    )


# ============================================================
# CLI
# ============================================================

def main():

    X, y = load_data()
    
    y = y.isin(
        ["PE", "IUGR+PE"]
    ).astype(int)   
    
    print(y)
    
    output_dir = create_output_directory(
        OUTPUT_DIR
    )

    run_analysis(
        X=X,
        y=y,
        output_dir=output_dir,
        folds=5,
        repeats=10,
        trees=500
    )


if __name__ == "__main__":
    main()