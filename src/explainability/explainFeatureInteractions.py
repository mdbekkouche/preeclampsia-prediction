"""
===========================================================
Feature Interaction Analysis
===========================================================

This module analyzes pairwise feature interactions of the
selected machine learning model using SHAP interaction values.

Outputs
-------
results/feature_interactions/

Author: Mohammed Bekkouche
"""

from pathlib import Path
import warnings

import joblib
import pandas as pd
import numpy as np
import shap

from sklearn.pipeline import Pipeline

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ===========================================================
# Configuration
# ===========================================================

CONFIG_FILE = "config/config.yaml"

MODEL_FILE = "results/models/best_model.pkl"

OUTPUT_DIR = Path("results/feature_interactions")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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


# ===========================================================
# Load Best Model
# ===========================================================

def load_best_model():
    """
    Load the trained model.
    """

    if not Path(MODEL_FILE).exists():

        raise FileNotFoundError(
            f"Cannot find model:\n{MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print(f"Loaded model:\n{MODEL_FILE}")

    return model


# ===========================================================
# Extract Estimator
# ===========================================================

def get_estimator(model):
    """
    Return the estimator if a Pipeline is used.
    """

    if isinstance(model, Pipeline):

        if "model" in model.named_steps:

            return model.named_steps["model"]

        return list(model.named_steps.values())[-1]

    return model


# ===========================================================
# Transform Features
# ===========================================================

def transform_features(model, X):
    """
    Apply preprocessing steps before the classifier.

    Returns
    -------
    ndarray
    """

    if not isinstance(model, Pipeline):

        return X

    Xt = X.copy()

    for name, step in model.named_steps.items():

        if name == "model":
            break

        if hasattr(step, "transform"):

            Xt = step.transform(Xt)

    return Xt


# ===========================================================
# Verify Tree-Based Model
# ===========================================================

def check_tree_model(model):
    """
    SHAP interaction values are only supported for tree models.
    """

    estimator = get_estimator(model)

    supported = (
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier"
    )

    model_name = estimator.__class__.__name__

    print(f"Model: {model_name}")

    if model_name not in supported:

        raise ValueError(

            "\nSHAP interaction values require a "
            "tree-based model.\n"
            f"Current model: {model_name}"

        )

    return estimator


# ===========================================================
# Create Tree Explainer
# ===========================================================

def create_tree_explainer(estimator):
    """
    Create TreeExplainer.
    """

    print("Creating TreeExplainer...")

    explainer = shap.TreeExplainer(
        estimator
    )

    return explainer


# ===========================================================
# Dataset Information
# ===========================================================

def print_dataset_info(X, y):

    print("=" * 60)

    print("Dataset")

    print("=" * 60)

    print(f"Samples : {len(X)}")

    print(f"Features: {X.shape[1]}")

    print("\nClass distribution")

    print(y.value_counts())

    print("=" * 60)

# ===========================================================
# Compute SHAP Interaction Values
# ===========================================================

def compute_interaction_values(
    explainer,
    X,
    feature_names
):
    """
    Compute SHAP interaction values.

    Parameters
    ----------
    explainer : shap.TreeExplainer

    X : ndarray or DataFrame

    feature_names : list

    Returns
    -------
    interaction_values : ndarray
        Shape = (n_samples, n_features, n_features)
    """

    print("\nComputing SHAP interaction values...")

    interaction_values = explainer.shap_interaction_values(X)

    # ---------------------------------------------------------
    # Binary classification
    # ---------------------------------------------------------

    if isinstance(interaction_values, list):

        # Older SHAP versions
        interaction_values = interaction_values[1]

    elif interaction_values.ndim == 4:

        # SHAP >= 0.45
        interaction_values = interaction_values[:, :, :, 1]

    print("Done.")

    print(f"Interaction tensor shape: {interaction_values.shape}")

    return interaction_values


# ===========================================================
# Mean Interaction Matrix
# ===========================================================

def compute_interaction_matrix(
    interaction_values,
    feature_names
):
    """
    Mean absolute interaction matrix.

    Returns
    -------
    DataFrame
    """

    print("Computing interaction matrix...")

    matrix = np.abs(
        interaction_values
    ).mean(axis=0)

    matrix_df = pd.DataFrame(

        matrix,

        index=feature_names,

        columns=feature_names

    )

    return matrix_df


# ===========================================================
# Save Interaction Matrix
# ===========================================================

def save_interaction_matrix(
    matrix_df
):

    filename = (
        OUTPUT_DIR
        / "interaction_matrix.csv"
    )

    matrix_df.to_csv(
        filename
    )

    print(f"Saved: {filename}")


# ===========================================================
# Interaction Ranking
# ===========================================================

def rank_interactions(
    matrix_df
):
    """
    Rank pairwise interactions.

    Returns
    -------
    DataFrame
    """

    print("Ranking interactions...")

    rows = []

    features = matrix_df.columns.tolist()

    for i in range(len(features)):

        for j in range(i + 1, len(features)):

            rows.append({

                "Feature1": features[i],

                "Feature2": features[j],

                "MeanInteraction":

                    matrix_df.iloc[i, j]

            })

    ranking = pd.DataFrame(rows)

    ranking = ranking.sort_values(

        "MeanInteraction",

        ascending=False

    )

    ranking.reset_index(

        drop=True,

        inplace=True

    )

    return ranking


# ===========================================================
# Save Ranking
# ===========================================================

def save_interaction_ranking(
    ranking
):

    filename = (
        OUTPUT_DIR
        / "interaction_ranking.csv"
    )

    ranking.to_csv(

        filename,

        index=False

    )

    print(f"Saved: {filename}")


# ===========================================================
# Top Interactions
# ===========================================================

def print_top_interactions(
    ranking,
    top_n=20
):
    """
    Print strongest interactions.
    """

    print("\n")
    print("=" * 70)
    print("Top Feature Interactions")
    print("=" * 70)

    for _, row in ranking.head(top_n).iterrows():

        print(

            f"{row.Feature1:<35}"

            f"{row.Feature2:<35}"

            f"{row.MeanInteraction:.5f}"

        )

    print("=" * 70)


# ===========================================================
# Interaction Summary Statistics
# ===========================================================

def interaction_statistics(
    ranking
):
    """
    Compute summary statistics.
    """

    stats = {

        "Number_of_Interactions":

            len(ranking),

        "Maximum":

            ranking["MeanInteraction"].max(),

        "Minimum":

            ranking["MeanInteraction"].min(),

        "Mean":

            ranking["MeanInteraction"].mean(),

        "Median":

            ranking["MeanInteraction"].median(),

        "Std":

            ranking["MeanInteraction"].std()

    }

    return pd.DataFrame([stats])


# ===========================================================
# Save Statistics
# ===========================================================

def save_statistics(
    stats
):

    filename = (

        OUTPUT_DIR

        / "interaction_summary.csv"

    )

    stats.to_csv(

        filename,

        index=False

    )

    print(f"Saved: {filename}")


# ===========================================================
# Extract Top Features
# ===========================================================

def top_interaction_features(
    ranking,
    n=10
):
    """
    Return unique features participating in the
    strongest interactions.
    """

    selected = []

    for _, row in ranking.iterrows():

        if row.Feature1 not in selected:

            selected.append(row.Feature1)

        if row.Feature2 not in selected:

            selected.append(row.Feature2)

        if len(selected) >= n:

            break

    return selected[:n]    


# ===========================================================
# Visualization
# ===========================================================

import matplotlib.pyplot as plt
import seaborn as sns


# ===========================================================
# Interaction Summary Plot
# ===========================================================

def save_interaction_summary_plot(
    interaction_values,
    X
):
    """
    SHAP interaction summary plot.
    """

    print("Generating interaction summary plot...")

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        interaction_values,
        X,
        show=False
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR /
        "interaction_summary.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ===========================================================
# Interaction Heatmap
# ===========================================================

def save_interaction_heatmap(
    matrix_df,
    top_n=20
):
    """
    Heatmap of strongest interactions.
    """

    print("Generating interaction heatmap...")

    features = (
        matrix_df.abs()
        .sum(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    matrix = matrix_df.loc[
        features,
        features
    ]

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrix,
        cmap="viridis",
        square=True,
        linewidths=0.5
    )

    plt.xticks(rotation=90)

    plt.yticks(rotation=0)

    plt.tight_layout()

    filename = (
        OUTPUT_DIR /
        "interaction_heatmap.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ===========================================================
# Top Interaction Bar Plot
# ===========================================================

def save_top_interactions(
    ranking,
    top_n=20
):

    print("Generating top interactions plot...")

    top = ranking.head(top_n).copy()

    top["Pair"] = (
        top["Feature1"]
        + "\n×\n"
        + top["Feature2"]
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top["Pair"],
        top["MeanInteraction"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Mean Absolute SHAP Interaction"
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR /
        "top_interactions.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ===========================================================
# Dependence Plot
# ===========================================================

import re

def safe_filename(text):
    """
    Convert any feature name into a valid filename.
    """

    text = str(text)

    # Replace forbidden Windows filename characters
    text = re.sub(r'[<>:"/\\|?*]', "_", text)

    # Remove brackets
    text = text.replace("[", "")
    text = text.replace("]", "")

    # Replace spaces
    text = text.replace(" ", "_")

    # Optional cleanup
    while "__" in text:
        text = text.replace("__", "_")

    return text

def save_dependence_plot(
    explanation,
    feature,
    interaction_feature
):
    """
    Dependence plot coloured by another feature.
    """

    print(
        f"Dependence: "
        f"{feature} × {interaction_feature}"
    )

    plt.figure(figsize=(7, 6))

    shap.plots.scatter(

        explanation[:, feature],

        color=explanation[:, interaction_feature],

        show=False

    )

    plt.tight_layout()

    #filename = (
    #    OUTPUT_DIR /
    #    f"{feature}_{interaction_feature}.png"
    #)
    
    filename = (
        OUTPUT_DIR /
        f"{safe_filename(feature)}_{safe_filename(interaction_feature)}.png"
    )
    
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ===========================================================
# Generate Dependence Plots
# ===========================================================

def save_top_dependence_plots(
    explanation,
    ranking,
    n=5
):
    """
    Generate dependence plots
    for strongest interactions.
    """

    for _, row in ranking.head(n).iterrows():

        save_dependence_plot(

            explanation,

            row.Feature1,

            row.Feature2

        )


# ===========================================================
# Network Table
# ===========================================================

def save_interaction_network(
    ranking,
    top_n=50
):
    """
    Save interaction network.
    """

    network = ranking.head(top_n)

    filename = (
        OUTPUT_DIR /
        "interaction_network.csv"
    )

    network.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")


# ===========================================================
# Generate All Figures
# ===========================================================

def generate_all_figures(

    interaction_values,

    interaction_matrix,

    interaction_ranking,

    explanation

):

    print("\n")
    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)

    save_interaction_summary_plot(

        interaction_values,

        explanation.data

    )

    save_interaction_heatmap(

        interaction_matrix

    )

    save_top_interactions(

        interaction_ranking

    )

    save_top_dependence_plots(

        explanation,

        interaction_ranking,

        n=5

    )

    save_interaction_network(

        interaction_ranking

    )

    print("=" * 60)
    print("Finished.")
    print("=" * 60)

# ===========================================================
# Part 3 - Visualization
# ===========================================================

import re
import matplotlib.pyplot as plt


# ===========================================================
# Safe Filename
# ===========================================================

def safe_filename(text):
    """
    Convert feature names into valid filenames.
    """

    text = str(text)

    text = re.sub(
        r'[\\/*?:"<>|]',
        "_",
        text
    )

    text = text.replace(" ", "_")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("(", "")
    text = text.replace(")", "")

    return text


# ===========================================================
# Interaction Heatmap
# ===========================================================

def save_interaction_heatmap(
    interaction_matrix,
    top_n=20
):
    """
    Heatmap of strongest interactions.
    """

    print(
        "\nGenerating interaction heatmap..."
    )

    feature_strength = (
        interaction_matrix.abs()
        .sum(axis=1)
        .sort_values(
            ascending=False
        )
    )

    selected_features = (
        feature_strength
        .head(top_n)
        .index
    )

    matrix = interaction_matrix.loc[
        selected_features,
        selected_features
    ]

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        matrix,
        aspect="auto"
    )

    plt.colorbar(
        label="Mean |Interaction|"
    )

    plt.xticks(
        range(len(matrix.columns)),
        matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(len(matrix.index)),
        matrix.index
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR
        / "interaction_heatmap.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {filename}"
    )


# ===========================================================
# Top Interactions Bar Chart
# ===========================================================

def save_top_interactions_plot(
    ranking,
    top_n=20
):
    """
    Plot strongest interactions.
    """

    print(
        "\nGenerating top interactions..."
    )

    top = ranking.head(top_n)

    labels = [

        f"{r.Feature1}\n×\n{r.Feature2}"

        for _, r in top.iterrows()

    ]

    values = top[
        "MeanInteraction"
    ]

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        labels,
        values
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Mean Absolute Interaction"
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR
        / "top_interactions.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {filename}"
    )


# ===========================================================
# Interaction Summary Matrix
# ===========================================================

def save_interaction_summary(
    interaction_matrix
):
    """
    Save matrix as image.
    """

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        interaction_matrix.values,
        aspect="auto"
    )

    plt.colorbar()

    plt.title(
        "Interaction Matrix"
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR
        / "interaction_matrix.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {filename}"
    )


# ===========================================================
# Plot Top Pair Matrix
# ===========================================================

def save_top_pair_matrix(
    interaction_matrix,
    ranking,
    top_n=15
):
    """
    Matrix restricted to top features.
    """

    selected = set()

    for _, row in ranking.head(top_n).iterrows():

        selected.add(
            row.Feature1
        )

        selected.add(
            row.Feature2
        )

    selected = list(selected)

    matrix = interaction_matrix.loc[
        selected,
        selected
    ]

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix.values,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(selected)),
        selected,
        rotation=90
    )

    plt.yticks(
        range(len(selected)),
        selected
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR
        / "top_feature_matrix.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {filename}"
    )


# ===========================================================
# Main
# ===========================================================

def main():

    print("=" * 70)
    print("FEATURE INTERACTION ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------
    # Load data
    # -------------------------------------------------------

    X, y = load_data()

    print_dataset_info(X, y)

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------

    model = load_best_model()

    # -------------------------------------------------------
    # Transform features
    # -------------------------------------------------------

    X_transformed = transform_features(
        model,
        X
    )

    # -------------------------------------------------------
    # Verify tree model
    # -------------------------------------------------------

    estimator = check_tree_model(
        model
    )

    # -------------------------------------------------------
    # Create TreeExplainer
    # -------------------------------------------------------

    explainer = create_tree_explainer(
        estimator
    )

    # -------------------------------------------------------
    # SHAP explanation
    # -------------------------------------------------------

    print("\nComputing SHAP explanation...")

    explanation = explainer(
        X_transformed
    )

    if len(explanation.values.shape) == 3:

        explanation = shap.Explanation(

            values=explanation.values[:, :, 1],

            base_values=explanation.base_values[:, 1],

            data=explanation.data,

            feature_names=X.columns.tolist()

        )

    print("Done.")

    # -------------------------------------------------------
    # Interaction values
    # -------------------------------------------------------

    interaction_values = compute_interaction_values(

        explainer,

        X_transformed,

        X.columns.tolist()

    )

    # -------------------------------------------------------
    # Mean interaction matrix
    # -------------------------------------------------------

    interaction_matrix = compute_interaction_matrix(

        interaction_values,

        X.columns.tolist()

    )

    save_interaction_matrix(
        interaction_matrix
    )

    # -------------------------------------------------------
    # Interaction ranking
    # -------------------------------------------------------

    interaction_ranking = rank_interactions(
        interaction_matrix
    )

    save_interaction_ranking(
        interaction_ranking
    )

    print_top_interactions(
        interaction_ranking,
        top_n=20
    )

    # -------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------

    stats = interaction_statistics(
        interaction_ranking
    )

    save_statistics(
        stats
    )

    # -------------------------------------------------------
    # Figures
    # -------------------------------------------------------

    generate_all_figures(

        interaction_values,

        interaction_matrix,

        interaction_ranking,

        explanation

    )

    # -------------------------------------------------------
    # Save interaction values
    # -------------------------------------------------------

    import joblib

    joblib.dump(

        interaction_values,

        OUTPUT_DIR / "interaction_values.pkl"

    )

    # -------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------

    report = OUTPUT_DIR / "report.md"

    with open(report, "w", encoding="utf-8") as f:

        f.write("# Feature Interaction Analysis\n\n")

        f.write(f"Samples: {len(X)}\n\n")

        f.write(f"Features: {X.shape[1]}\n\n")

        f.write("## Top 20 Interactions\n\n")

        for _, row in interaction_ranking.head(20).iterrows():

            f.write(

                f"- {row.Feature1} × "
                f"{row.Feature2}: "
                f"{row.MeanInteraction:.5f}\n"

            )

        f.write("\n")

        f.write("Generated files:\n")

        f.write("- interaction_summary.png\n")

        f.write("- interaction_heatmap.png\n")

        f.write("- top_interactions.png\n")

        f.write("- interaction_matrix.csv\n")

        f.write("- interaction_ranking.csv\n")

        f.write("- interaction_summary.csv\n")

    print(f"\nSaved: {report}")

    print("\n")
    print("=" * 70)
    print("FEATURE INTERACTION ANALYSIS COMPLETED")
    print("=" * 70)


# ===========================================================
# Entry point
# ===========================================================

if __name__ == "__main__":
    main()    