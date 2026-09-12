#!/usr/bin/env python3

"""
SHAP Interaction Stability Across Cross-Validation Folds

Purpose
-------
Assess the stability of pairwise SHAP interactions for a
Random Forest classifier across cross-validation splits.

For each split:
    1. Train Random Forest on training data.
    2. Calculate SHAP interaction values on validation data.
    3. Compute mean absolute interaction strength for each
       feature pair.
    4. Rank feature pairs.

Across splits:
    5. Compute mean, SD, CV, rank statistics and recurrence.
    6. Compute rank correlations across splits.
    7. Compute Jaccard similarity for top interaction sets.
    8. Generate publication-ready CSV files and figures.

Outputs
-------
results/shap-interaction-stability/
    shap_interaction_split_values.csv
    shap_interaction_stability_summary.csv
    shap_interaction_split_ranks.csv
    shap_interaction_rank_correlation.csv
    shap_interaction_jaccard.csv
    shap_interaction_heatmap.csv
    shap_interaction_report.txt
    shap_interaction_stability.png
    shap_interaction_heatmap.png
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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score


warnings.filterwarnings("ignore")


# Configuration
# ===========================================================

CONFIG_FILE = "config/config.yaml"

MODEL_FILE = "results/models/best_model.pkl"

OUTPUT_DIR = Path("results/shap-interaction-stability")

# ============================================================
# General configuration
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


# ============================================================
# SHAP interaction extraction
# ============================================================

def extract_binary_interactions(
    model,
    X_valid: pd.DataFrame
) -> np.ndarray:

    """
    Return a matrix:

        samples x features x features

    containing SHAP interaction values for the positive class.
    """

    explainer = shap.TreeExplainer(
        model
    )

    interaction_values = (
        explainer.shap_interaction_values(
            X_valid
        )
    )

    # --------------------------------------------------------
    # Older SHAP versions
    #
    # Binary classification:
    # [
    #     class_0_matrix,
    #     class_1_matrix
    # ]
    # --------------------------------------------------------

    if isinstance(
        interaction_values,
        list
    ):

        if len(interaction_values) != 2:
            raise ValueError(
                "Unexpected number of SHAP outputs."
            )

        values = np.asarray(
            interaction_values[1]
        )

    else:

        values = np.asarray(
            interaction_values
        )

    # --------------------------------------------------------
    # Shape handling for different SHAP versions
    # --------------------------------------------------------

    if values.ndim == 3:

        # samples x features x features
        return values

    if values.ndim == 4:

        # Possible forms:
        #
        # samples x features x features x classes
        # or
        # classes x samples x features x features

        if values.shape[-1] == 2:

            return values[:, :, :, 1]

        if values.shape[0] == 2:

            return values[1]

    raise ValueError(
        f"Unexpected SHAP interaction shape: "
        f"{values.shape}"
    )


# ============================================================
# Interaction summary
# ============================================================

def mean_absolute_pairwise_interactions(
    interaction_values: np.ndarray,
    feature_names: list[str]
) -> pd.Series:

    """
    Calculate mean absolute SHAP interaction strength
    for each unique feature pair.

    Diagonal terms are excluded because they represent
    main effects rather than pairwise interactions.
    """

    n_samples, n_features, _ = (
        interaction_values.shape
    )

    values = np.abs(
        interaction_values
    )

    result = {}

    for i, j in combinations(
        range(n_features),
        2
    ):

        pair_values = values[
            :,
            i,
            j
        ]

        result[
            f"{feature_names[i]} × "
            f"{feature_names[j]}"
        ] = np.mean(
            pair_values
        )

    return pd.Series(
        result,
        dtype=float
    )


# ============================================================
# Ranking
# ============================================================

def rank_series(
    values: pd.Series
) -> pd.Series:

    return values.rank(
        ascending=False,
        method="average"
    )


# ============================================================
# Jaccard similarity
# ============================================================

def calculate_jaccard_similarity(
    rank_df: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:

    splits = list(
        rank_df.columns
    )

    rows = []

    for i in range(
        len(splits)
    ):

        set_a = set(
            rank_df.index[
                rank_df[splits[i]] <= top_k
            ]
        )

        for j in range(
            i + 1,
            len(splits)
        ):

            set_b = set(
                rank_df.index[
                    rank_df[splits[j]] <= top_k
                ]
            )

            intersection = (
                len(set_a & set_b)
            )

            union = (
                len(set_a | set_b)
            )

            jaccard = (
                intersection / union
                if union > 0
                else np.nan
            )

            rows.append({
                "split_1": splits[i],
                "split_2": splits[j],
                "top_k": top_k,
                "intersection": intersection,
                "union": union,
                "jaccard_similarity": jaccard
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Rank correlations
# ============================================================

def calculate_rank_correlations(
    rank_df: pd.DataFrame
) -> pd.DataFrame:

    splits = list(
        rank_df.columns
    )

    rows = []

    for i in range(
        len(splits)
    ):

        for j in range(
            i + 1,
            len(splits)
        ):

            rho, p_value = spearmanr(
                rank_df[splits[i]],
                rank_df[splits[j]]
            )

            rows.append({
                "split_1": splits[i],
                "split_2": splits[j],
                "spearman_rho": rho,
                "p_value": p_value
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plotting
# ============================================================

def plot_interaction_stability(
    summary_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20
) -> None:

    data = (
        summary_df
        .sort_values(
            "mean_shap_interaction",
            ascending=False
        )
        .head(top_n)
        .sort_values(
            "mean_shap_interaction"
        )
    )

    plt.figure(
        figsize=(11, 9)
    )

    plt.barh(
        data.index,
        data["mean_shap_interaction"],
        xerr=data["std_shap_interaction"],
        capsize=3,
        color='#007000'
    )
    
    

    plt.xlabel(
        "Mean absolute SHAP interaction value ± SD"
    )

    plt.ylabel(
        "Feature pair"
    )

    plt.title(
        "SHAP Interaction Stability Across Cross-Validation Splits"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_interaction_heatmap(
    interaction_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10
) -> None:

    top_pairs = (
        interaction_df
        .sort_values(
            ascending=False
        )
        .head(top_n)
    )

    pairs = list(
        top_pairs.index
    )

    features = set()

    for pair in pairs:

        left, right = pair.split(
            " × ",
            maxsplit=1
        )

        features.add(left)
        features.add(right)

    features = sorted(
        features
    )

    matrix = pd.DataFrame(
        0.0,
        index=features,
        columns=features
    )

    for pair in pairs:

        left, right = pair.split(
            " × ",
            maxsplit=1
        )

        value = top_pairs.loc[
            pair
        ]

        matrix.loc[
            left,
            right
        ] = value

        matrix.loc[
            right,
            left
        ] = value

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix.values,
        aspect="auto"
    )

    plt.xticks(
        range(len(features)),
        features,
        rotation=90
    )

    plt.yticks(
        range(len(features)),
        features
    )

    plt.colorbar(
        label="Mean absolute SHAP interaction"
    )

    plt.title(
        "Mean SHAP Interaction Strength"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main analysis
# ============================================================

def run_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    folds: int = 5,
    repeats: int = 1,
    trees: int = 500,
    top_k: int = 10
) -> None:

    feature_names = list(
        X.columns
    )

    print("\n" + "=" * 70)
    print("SHAP INTERACTION STABILITY ANALYSIS")
    print("=" * 70)

    print(
        f"Samples:       {len(X)}"
    )

    print(
        f"Features:      {X.shape[1]}"
    )

    print(
        f"CV folds:      {folds}"
    )

    print(
        f"Repeats:       {repeats}"
    )

    print(
        f"RF trees:      {trees}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    if repeats == 1:

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_STATE
        )

    else:

        cv = RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=RANDOM_STATE
        )

    all_interactions = {}

    split_auc = []
    split_f1 = []

    # --------------------------------------------------------
    # CV loop
    # --------------------------------------------------------

    for split_number, (
        train_idx,
        valid_idx
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        repeat_number = (
            (split_number - 1) // folds
        ) + 1

        fold_number = (
            (split_number - 1) % folds
        ) + 1

        split_name = (
            f"repeat_{repeat_number:02d}_"
            f"fold_{fold_number:02d}"
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
                RANDOM_STATE + split_number
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
        # Predictive performance
        # ----------------------------------------------------

        probability = (
            model.predict_proba(X_valid)[:, 1]
        )

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

        split_auc.append(auc)
        split_f1.append(f1)

        print(
            f"ROC-AUC = {auc:.4f} | "
            f"F1 = {f1:.4f}"
        )

        # ----------------------------------------------------
        # SHAP interaction values
        # ----------------------------------------------------

        interaction_values = (
            extract_binary_interactions(
                model,
                X_valid
            )
        )

        print(
            "SHAP interaction shape:",
            interaction_values.shape
        )

        # ----------------------------------------------------
        # Mean absolute interaction per pair
        # ----------------------------------------------------

        interaction_series = (
            mean_absolute_pairwise_interactions(
                interaction_values,
                feature_names
            )
        )

        all_interactions[
            split_name
        ] = interaction_series

        print(
            "Top interactions:"
        )

        print(
            interaction_series
            .sort_values(
                ascending=False
            )
            .head(5)
        )

    # ========================================================
    # Combine split results
    # ========================================================

    interaction_df = pd.DataFrame(
        all_interactions
    )

    rank_df = interaction_df.apply(
        rank_series,
        axis=0
    )

    # ========================================================
    # Interaction stability summary
    # ========================================================

    summary = pd.DataFrame(
        index=interaction_df.index
    )

    summary[
        "mean_shap_interaction"
    ] = interaction_df.mean(
        axis=1
    )

    summary[
        "std_shap_interaction"
    ] = interaction_df.std(
        axis=1
    )

    summary[
        "cv_shap_interaction"
    ] = (
        summary["std_shap_interaction"]
        /
        summary["mean_shap_interaction"].replace(
            0,
            np.nan
        )
    )

    summary[
        "mean_rank"
    ] = rank_df.mean(
        axis=1
    )

    summary[
        "std_rank"
    ] = rank_df.std(
        axis=1
    )

    summary[
        "min_rank"
    ] = rank_df.min(
        axis=1
    )

    summary[
        "max_rank"
    ] = rank_df.max(
        axis=1
    )

    summary[
        "rank_range"
    ] = (
        summary["max_rank"]
        -
        summary["min_rank"]
    )

    summary[
        "top10_frequency"
    ] = (
        rank_df <= min(
            top_k,
            len(summary)
        )
    ).mean(
        axis=1
    )

    summary[
        "top20_frequency"
    ] = (
        rank_df <= min(
            20,
            len(summary)
        )
    ).mean(
        axis=1
    )

    # --------------------------------------------------------
    # A simple reproducibility score
    #
    # Higher = stronger recurrence + smaller rank variation
    # --------------------------------------------------------

    summary[
        "interaction_stability_score"
    ] = (
        summary["top10_frequency"]
        /
        (
            1.0
            +
            summary["std_rank"]
        )
    )

    summary = summary.sort_values(
        "mean_shap_interaction",
        ascending=False
    )

    # ========================================================
    # Rank correlation
    # ========================================================

    correlation_df = (
        calculate_rank_correlations(
            rank_df
        )
    )

    # ========================================================
    # Top-k Jaccard similarity
    # ========================================================

    jaccard_df = (
        calculate_jaccard_similarity(
            rank_df,
            top_k=top_k
        )
    )

    # ========================================================
    # Save CSV files
    # ========================================================

    interaction_df.to_csv(
        output_dir /
        "shap_interaction_split_values.csv"
    )

    summary.to_csv(
        output_dir /
        "shap_interaction_stability_summary.csv"
    )

    rank_df.to_csv(
        output_dir /
        "shap_interaction_split_ranks.csv"
    )

    correlation_df.to_csv(
        output_dir /
        "shap_interaction_rank_correlation.csv",
        index=False
    )

    jaccard_df.to_csv(
        output_dir /
        "shap_interaction_jaccard.csv",
        index=False
    )

    # ========================================================
    # Heatmap data
    # ========================================================

    mean_interactions = (
        interaction_df.mean(
            axis=1
        )
    )

    mean_interactions.to_csv(
        output_dir /
        "shap_interaction_heatmap.csv"
    )

    # ========================================================
    # Figures
    # ========================================================

    plot_interaction_stability(
        summary,
        output_dir /
        "shap_interaction_stability.png"
    )

    plot_interaction_heatmap(
        mean_interactions,
        output_dir /
        "shap_interaction_heatmap.png"
    )

    # ========================================================
    # Report
    # ========================================================

    report_path = (
        output_dir /
        "shap_interaction_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "SHAP Interaction Stability Analysis\n"
        )

        report.write(
            "=" * 60 + "\n\n"
        )

        report.write(
            f"Samples: {len(X)}\n"
        )

        report.write(
            f"Features: {X.shape[1]}\n"
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
            f"{np.mean(split_auc):.4f}\n"
        )

        report.write(
            f"SD ROC-AUC: "
            f"{np.std(split_auc, ddof=1):.4f}\n"
        )

        report.write(
            f"Mean F1: "
            f"{np.mean(split_f1):.4f}\n"
        )

        report.write(
            f"SD F1: "
            f"{np.std(split_f1, ddof=1):.4f}\n\n"
        )

        if not correlation_df.empty:

            report.write(
                "Interaction rank stability\n"
            )

            report.write(
                "-" * 40 + "\n"
            )

            report.write(
                f"Mean Spearman rho: "
                f"{correlation_df['spearman_rho'].mean():.4f}\n"
            )

            report.write(
                f"SD Spearman rho: "
                f"{correlation_df['spearman_rho'].std():.4f}\n\n"
            )

        if not jaccard_df.empty:

            report.write(
                "Top interaction recurrence\n"
            )

            report.write(
                "-" * 40 + "\n"
            )

            report.write(
                f"Mean Jaccard similarity "
                f"(top-{top_k}): "
                f"{jaccard_df['jaccard_similarity'].mean():.4f}\n\n"
            )

        report.write(
            "Top 20 interactions\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        for pair, row in summary.head(20).iterrows():

            report.write(
                f"{pair}\n"
                f"  Mean interaction = "
                f"{row['mean_shap_interaction']:.6f}\n"
                f"  SD = "
                f"{row['std_shap_interaction']:.6f}\n"
                f"  Mean rank = "
                f"{row['mean_rank']:.2f}\n"
                f"  Rank range = "
                f"{row['rank_range']:.0f}\n"
                f"  Top-10 frequency = "
                f"{row['top10_frequency']:.2f}\n\n"
            )

    # ========================================================
    # Console summary
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP SHAP INTERACTIONS")
    print("=" * 70)

    print(
        summary.head(15)[
            [
                "mean_shap_interaction",
                "std_shap_interaction",
                "mean_rank",
                "rank_range",
                "top10_frequency"
            ]
        ]
    )

    print("\nResults saved to:")
    print(
        output_dir.resolve()
    )


# ============================================================
# Command-line interface
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
        repeats=1,
        trees=500,
        top_k=20
    )


if __name__ == "__main__":
    main()