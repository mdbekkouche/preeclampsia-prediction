"""
===========================================================
SHAP Heatmap Visualization
===========================================================

This script loads the best trained model, computes SHAP values,
and generates a SHAP heatmap.

Outputs
-------
results/shap/
    shap_heatmap.png

Author: Mohammed Bekkouche
"""

from pathlib import Path
import warnings

import joblib
import shap
import matplotlib.pyplot as plt

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CONFIG_FILE = "config/config.yaml"

MODEL_FILE = "results/models/best_model.pkl"

OUTPUT_DIR = Path("results/shap")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_data():

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


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

def load_model():

    print("Loading best model...")

    model = joblib.load(MODEL_FILE)

    return model


# ------------------------------------------------------------
# Transform data if Pipeline
# ------------------------------------------------------------

def transform_features(model, X):

    try:

        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):

            Xt = X.copy()

            for name, step in model.named_steps.items():

                if name == "model":
                    break

                Xt = step.transform(Xt)

            return Xt

    except Exception:
        pass

    return X


# ------------------------------------------------------------
# Create SHAP explainer
# ------------------------------------------------------------

def create_explainer(model, X):

    estimator = model

    try:

        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):

            estimator = model.named_steps["model"]

    except Exception:
        pass

    model_name = estimator.__class__.__name__

    print(f"Model: {model_name}")

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
    )

    if model_name in tree_models:

        return shap.TreeExplainer(estimator)

    elif model_name in linear_models:

        return shap.LinearExplainer(
            estimator,
            X
        )

    else:

        background = shap.sample(
            X,
            min(50, len(X)),
            random_state=42
        )

        if hasattr(model, "predict_proba"):

            return shap.KernelExplainer(
                model.predict_proba,
                background
            )

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


# ------------------------------------------------------------
# Compute explanation
# ------------------------------------------------------------

def compute_explanation(
    explainer,
    X
):

    print("Computing SHAP values...")

    explanation = explainer(X)

    if len(explanation.values.shape) == 3:

        explanation = shap.Explanation(

            values=explanation.values[:, :, 1],

            base_values=explanation.base_values[:, 1],

            data=explanation.data,

            feature_names=explanation.feature_names

        )

    return explanation


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

def generate_heatmap(
    explanation
):

    print("Generating SHAP heatmap...")

    plt.figure(figsize=(12, 10))

    shap.plots.heatmap(

        explanation,

        max_display=25,

        show=False

    )

    plt.tight_layout()

    filename = OUTPUT_DIR / "shap_heatmap.png"

    plt.savefig(

        filename,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(f"Saved: {filename}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("SHAP HEATMAP")
    print("=" * 60)

    X, y = load_data()

    model = load_model()

    X_transformed = transform_features(
        model,
        X
    )

    explainer = create_explainer(
        model,
        X_transformed
    )

    explanation = compute_explanation(
        explainer,
        X_transformed
    )

    generate_heatmap(
        explanation
    )

    print("\nDone.")


if __name__ == "__main__":
    main()