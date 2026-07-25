"""
===========================================================
SHAP Analysis for the Best Preeclampsia Prediction Model
===========================================================

This module explains the predictions of the selected best
machine learning model using SHAP.

Outputs
-------
results/shap/
    feature_importance.csv
    shap_summary_bar.png
    shap_summary_beeswarm.png
    dependence_<feature>.png
    waterfall_<patient>.png

Author: Mohammed Bekkouche
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CONFIG_FILE = "config/config.yaml"

MODEL_FILE = "results/models/best_model.pkl"

OUTPUT_DIR = Path("results/shap")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------

def load_data():
    """
    Load the dataset and build engineered features.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target vector.
    """

    config = load_config(CONFIG_FILE)

    df = load_dataset(config["data"]["raw_file"])

    df = build_interaction_features(df, config)

    target = config["target"]["column"]

    features = (
        config["features"]["clinical"]
        + config["features"]["doppler"]
        + config["features"]["biomarkers"]
        + config["features"]["interaction_features"]
    )

    features = [f for f in features if f in df.columns]

    X = df[features]

    y = df[target]
    
    y = y.isin(
            ["PE", "IUGR+PE"]
        ).astype(int)

    return X, y


# ---------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------

def load_best_model():
    """
    Load the best trained model from disk.
    """

    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print(f"Loaded model: {MODEL_FILE}")

    return model


# ---------------------------------------------------------------------
# Extract estimator from Pipeline
# ---------------------------------------------------------------------

def get_estimator(model):
    """
    Returns the fitted estimator.

    Parameters
    ----------
    model : sklearn Pipeline or estimator

    Returns
    -------
    estimator
    """

    if isinstance(model, Pipeline):
        return model.named_steps["model"]

    return model


# ---------------------------------------------------------------------
# Transform features using preprocessing steps
# ---------------------------------------------------------------------

def transform_features(model, X):
    """
    Apply all preprocessing steps except the classifier.

    This is important because SHAP should explain the exact
    features seen by the estimator.

    Returns
    -------
    X_transformed
    """

    if not isinstance(model, Pipeline):
        return X

    X_transformed = X.copy()

    for name, step in model.named_steps.items():

        if name == "model":
            break

        if hasattr(step, "transform"):
            X_transformed = step.transform(X_transformed)

    return X_transformed


# ---------------------------------------------------------------------
# Create SHAP explainer
# ---------------------------------------------------------------------

def create_explainer(model, X_background):
    """
    Automatically select the appropriate SHAP explainer.

    Parameters
    ----------
    model : fitted estimator

    X_background : ndarray

    Returns
    -------
    shap explainer
    """

    estimator = get_estimator(model)

    tree_models = (
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
    )

    linear_models = (
        "LogisticRegression",
        "LinearRegression",
    )

    model_name = estimator.__class__.__name__

    print(f"Using SHAP for {model_name}")

    if model_name in tree_models:

        return shap.TreeExplainer(estimator)

    elif model_name in linear_models:

        return shap.LinearExplainer(
            estimator,
            X_background,
        )

    else:

        print("Falling back to KernelExplainer.")

        if hasattr(model, "predict_proba"):

            background = shap.sample(
                pd.DataFrame(X_background),
                min(50, len(X_background)),
                random_state=42,
            )

            return shap.KernelExplainer(
                model.predict_proba,
                background,
            )

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def print_dataset_info(X, y):

    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(f"Samples  : {len(X)}")
    print(f"Features : {X.shape[1]}")

    print("\nFeature names:")

    for feature in X.columns:
        print(f"  • {feature}")

    print("\nClass distribution")

    print(y.value_counts())

    print("=" * 60)
    
# Part 2

# ---------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------

def compute_shap_values(model, explainer, X):
    """
    Compute SHAP values for the dataset.

    Parameters
    ----------
    model : fitted model or Pipeline

    explainer : SHAP explainer

    X : pandas.DataFrame

    Returns
    -------
    shap_values : ndarray
        SHAP values for the positive class.

    X_explain : ndarray
        Feature matrix used by SHAP.
    """

    X_explain = transform_features(model, X)

    print("\nComputing SHAP values...")
    print(f"Samples : {len(X)}")
    print(f"Features: {X.shape[1]}")

    # -------------------------------------------------------------
    # TreeExplainer / LinearExplainer
    # -------------------------------------------------------------
    try:

        explanation = explainer(X_explain)

        shap_values = explanation.values

    except Exception:

        # ---------------------------------------------------------
        # KernelExplainer
        # ---------------------------------------------------------

        shap_values = explainer.shap_values(X_explain)

    # -------------------------------------------------------------
    # Binary classification
    # -------------------------------------------------------------
    if isinstance(shap_values, list):

        # Older SHAP versions
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    elif len(shap_values.shape) == 3:

        # SHAP >= 0.45
        shap_values = shap_values[:, :, 1]

    print("Done.")

    explanation = explainer(X_explain)

    return explanation


# ---------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------

def compute_feature_importance(shap_values, feature_names):
    """
    Compute mean absolute SHAP values.

    Parameters
    ----------
    shap_values : ndarray

    feature_names : list

    Returns
    -------
    DataFrame
    """

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": importance
    })

    importance_df = importance_df.sort_values(
        by="MeanAbsSHAP",
        ascending=False
    )

    return importance_df


# ---------------------------------------------------------------------
# Save feature importance
# ---------------------------------------------------------------------

def save_feature_importance(importance_df):
    """
    Save feature importance as CSV.
    """

    filename = OUTPUT_DIR / "feature_importance.csv"

    importance_df.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")

    print("\nTop 10 features")

    print(importance_df.head(10))


# ---------------------------------------------------------------------
# Print feature importance
# ---------------------------------------------------------------------

def print_feature_importance(importance_df, n=20):
    """
    Pretty-print top features.
    """

    print("\n")
    print("=" * 60)
    print("Global SHAP Feature Importance")
    print("=" * 60)

    for _, row in importance_df.head(n).iterrows():

        print(
            f"{row.Feature:<40}"
            f"{row.MeanAbsSHAP:.5f}"
        )

    print("=" * 60)


# ---------------------------------------------------------------------
# Select representative patients
# ---------------------------------------------------------------------

def select_representative_patients(
    model,
    X,
    y,
    n_positive=2,
    n_negative=2
):
    """
    Select representative patients for
    local SHAP explanations.

    Returns
    -------
    list of indices
    """

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[:, 1]

    else:

        probabilities = model.decision_function(X)

    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]

    selected = []

    if len(positives):

        positive_rank = positives[
            np.argsort(probabilities[positives])[::-1]
        ]

        selected.extend(
            positive_rank[:n_positive]
        )

    if len(negatives):

        negative_rank = negatives[
            np.argsort(probabilities[negatives])
        ]

        selected.extend(
            negative_rank[:n_negative]
        )

    print(
        "\nRepresentative patients:",
        selected
    )

    return selected


# ---------------------------------------------------------------------
# Save SHAP values
# ---------------------------------------------------------------------

def save_shap_values(
    shap_values,
    feature_names
):
    """
    Save raw SHAP values.

    Useful for later stability analysis.
    """

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )

    filename = OUTPUT_DIR / "shap_values.csv"

    shap_df.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

def compute_summary_statistics(shap_values):
    """
    Compute descriptive statistics
    of SHAP values.
    """

    stats = pd.DataFrame({
        "Mean": np.mean(shap_values, axis=0),
        "Std": np.std(shap_values, axis=0),
        "Median": np.median(shap_values, axis=0),
        "Min": np.min(shap_values, axis=0),
        "Max": np.max(shap_values, axis=0),
    })

    return stats    
    
# Part 3
# ---------------------------------------------------------------------
# SHAP Visualization Functions
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import shap


def save_bar_plot(explanation):
    """
    Global feature importance (bar plot).
    """

    print("Generating SHAP bar plot...")

    plt.figure(figsize=(8, 6))

    shap.plots.bar(
        explanation,
        show=False,
        max_display=20
    )

    plt.tight_layout()

    filename = OUTPUT_DIR / "shap_summary_bar.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ---------------------------------------------------------------------

def save_beeswarm_plot(explanation):
    """
    SHAP beeswarm plot.
    """

    print("Generating SHAP beeswarm plot...")

    plt.figure(figsize=(10, 7))

    shap.plots.beeswarm(
        explanation,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    filename = OUTPUT_DIR / "shap_summary_beeswarm.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ---------------------------------------------------------------------

def save_waterfall_plots(
    explanation,
    patient_indices
):
    """
    Generate local explanations.
    """

    print("Generating waterfall plots...")

    for idx in patient_indices:

        plt.figure(figsize=(8, 6))

        shap.plots.waterfall(
            explanation[idx],
            max_display=15,
            show=False
        )

        plt.tight_layout()

        filename = OUTPUT_DIR / f"waterfall_patient_{idx}.png"

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {filename}")


from pathlib import Path
import re


def safe_path(directory: Path, prefix: str, feature: str, ext="png"):

    feature = re.sub(r'[\\/*?:"<>|]', "_", feature)
    feature = feature.replace(" ", "_")
    feature = feature.replace("[", "")
    feature = feature.replace("]", "")

    return directory / f"{prefix}_{feature}.{ext}"

# ---------------------------------------------------------------------

def save_dependence_plots(
    explanation,
    importance_df,
    max_features=5
):
    """
    Generate dependence plots for the most important features.
    """

    print("Generating dependence plots...")

    top_features = importance_df.head(max_features)["Feature"]

    for feature in top_features:

        plt.figure(figsize=(7, 5))

        shap.plots.scatter(
            explanation[:, feature],
            show=False
        )

        plt.tight_layout()

        filename = safe_path(
            OUTPUT_DIR,
            "dependence",
            feature
        )
        #filename = (
        #    OUTPUT_DIR /
        #    f"dependence_{feature}.png"
        #)

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {filename}")


# ---------------------------------------------------------------------

def save_decision_plot(
    explanation,
    sample_index=0
):
    """
    Decision plot for one patient.

    Works for tree and linear explainers.
    """

    print("Generating decision plot...")

    try:

        plt.figure(figsize=(9, 5))

        shap.decision_plot(
            explanation.base_values[sample_index],
            explanation.values[sample_index],
            feature_names=explanation.feature_names,
            show=False
        )

        plt.tight_layout()

        filename = OUTPUT_DIR / "decision_plot.png"

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {filename}")

    except Exception as e:

        print(f"Decision plot skipped ({e})")


# ---------------------------------------------------------------------

def generate_all_plots(
    explanation,
    importance_df,
    patient_indices
):
    """
    Generate all SHAP visualizations.
    """

    print("\n")
    print("=" * 60)
    print("Generating SHAP Figures")
    print("=" * 60)

    save_bar_plot(explanation)

    save_beeswarm_plot(explanation)

    save_dependence_plots(
        explanation,
        importance_df,
        max_features=5
    )

    save_waterfall_plots(
        explanation,
        patient_indices
    )

    save_decision_plot(
        explanation,
        sample_index=patient_indices[0]
    )

    print("=" * 60)
    print("All SHAP figures generated.")
    print("=" * 60)

# Part 4

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Complete SHAP analysis pipeline.
    """

    print("=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    X, y = load_data()

    print_dataset_info(X, y)

    # -------------------------------------------------------------
    # Load trained model
    # -------------------------------------------------------------
    model = load_best_model()

    # -------------------------------------------------------------
    # Transform features
    # -------------------------------------------------------------
    X_explain = transform_features(model, X)

    # -------------------------------------------------------------
    # Build SHAP explainer
    # -------------------------------------------------------------
    explainer = create_explainer(
        model,
        X_explain
    )

    # -------------------------------------------------------------
    # Compute SHAP explanation
    # -------------------------------------------------------------
    print("\nComputing SHAP explanation...")

    explanation = explainer(X_explain)

    # Binary classification
    if len(explanation.values.shape) == 3:
        explanation = shap.Explanation(
            values=explanation.values[:, :, 1],
            base_values=explanation.base_values[:, 1],
            data=explanation.data,
            feature_names=explanation.feature_names
        )

    print("Done.")

    # -------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------
    importance_df = compute_feature_importance(
        explanation.values,
        explanation.feature_names
    )

    save_feature_importance(
        importance_df
    )

    print_feature_importance(
        importance_df,
        n=20
    )

    # -------------------------------------------------------------
    # Save raw SHAP values
    # -------------------------------------------------------------
    save_shap_values(
        explanation.values,
        explanation.feature_names
    )

    # -------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------
    stats = compute_summary_statistics(
        explanation.values
    )

    stats.index = explanation.feature_names

    stats.to_csv(
        OUTPUT_DIR / "shap_statistics.csv"
    )

    print("Saved: shap_statistics.csv")

    # -------------------------------------------------------------
    # Representative patients
    # -------------------------------------------------------------
    patient_indices = select_representative_patients(
        model,
        X,
        y,
        n_positive=2,
        n_negative=2
    )

    # -------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------
    generate_all_plots(
        explanation,
        importance_df,
        patient_indices
    )

    # -------------------------------------------------------------
    # Save explanation object
    # -------------------------------------------------------------
    import joblib

    joblib.dump(
        explanation,
        OUTPUT_DIR / "shap_explanation.pkl"
    )

    print("Saved: shap_explanation.pkl")

    # -------------------------------------------------------------
    # Save top features
    # -------------------------------------------------------------
    top10 = importance_df.head(10)

    top10.to_csv(
        OUTPUT_DIR / "top10_features.csv",
        index=False
    )

    # -------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------
    report_file = OUTPUT_DIR / "report.md"

    with open(report_file, "w") as f:

        f.write("# SHAP Analysis Report\n\n")

        f.write(f"Samples: {len(X)}\n\n")

        f.write(f"Features: {X.shape[1]}\n\n")

        f.write("## Top 10 Features\n\n")

        f.write(top10.to_markdown(index=False))

        f.write("\n\n")

        f.write("Generated Figures\n")

        f.write("- Summary Bar Plot\n")
        f.write("- Beeswarm Plot\n")
        f.write("- Dependence Plots\n")
        f.write("- Waterfall Plots\n")
        f.write("- Decision Plot\n")

    print(f"Saved: {report_file}")

    print("\n")
    print("=" * 70)
    print("SHAP analysis completed successfully.")
    print("=" * 70)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()    