"""
==============================================================
Gestational Age at Delivery Prediction (Regression)
==============================================================

Predict:
    Gestational age at delivery [weeks]

using

    - Clinical features
    - Doppler features
    - Biomarkers
    - Engineered interaction features

Author:
    Mohammed Bekkouche
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import joblib

from src.data.load_data import load_dataset
from src.features.build_features import (
    build_interaction_features
)
from src.utils.config import load_config


warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================

CONFIG_FILE = "config/config.yaml"

OUTPUT_DIR = Path("results")

MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "Gestational age at delivery [weeks]"


# ==========================================================
# Load Dataset
# ==========================================================

def load_data():

    config = load_config(CONFIG_FILE)

    df = load_dataset(
        config["data"]["raw_file"]
    )
    df = df.drop(columns=["Birth weight [g]"])
    df = df.drop(columns=["PE_Label"])
    
    df = build_interaction_features(
        df,
        config
    )

    feature_columns = (

        config["features"]["clinical"]

        + config["features"]["doppler"]

        + config["features"]["biomarkers"]

        + config["features"]["interaction_features"]

    )

    feature_columns = [

        f for f in feature_columns

        if f in df.columns

    ]

    X = df[feature_columns].copy()

    y = df[TARGET].copy()

    return X, y


# ==========================================================
# Preprocessing
# ==========================================================

def create_preprocessor(features):

    numeric_transformer = Pipeline(

        steps=[

            (

                "imputer",

                SimpleImputer(

                    strategy="median"

                ),

            ),

            (

                "scaler",

                StandardScaler(),

            ),

        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "num",

                numeric_transformer,

                features,

            )

        ]

    )

    return preprocessor


# ==========================================================
# Regression Models
# ==========================================================

def create_models():

    models = {

        "LinearRegression":

            LinearRegression(),

        "Ridge":

            Ridge(

                alpha=1.0,

                random_state=42,

            ),

        "Lasso":

            Lasso(
                alpha=0.01,
                max_iter=10000,
                random_state=42,
            ),

        "ElasticNet":

            ElasticNet(

                alpha=0.01,

                l1_ratio=0.5,

                random_state=42,

            ),

        "RandomForest":

            RandomForestRegressor(

                n_estimators=300,

                random_state=42,

                n_jobs=-1,

            ),

        "ExtraTrees":

            ExtraTreesRegressor(

                n_estimators=300,

                random_state=42,

                n_jobs=-1,

            ),

        "GradientBoosting":

            GradientBoostingRegressor(

                random_state=42,

            ),

    }

    return models


# ==========================================================
# Cross Validation
# ==========================================================

def create_cv():

    return RepeatedKFold(

        n_splits=5,

        n_repeats=10,

        random_state=42,

    )


# ==========================================================
# Regression Metrics
# ==========================================================

def rmse(

    y_true,

    y_pred,

):

    return np.sqrt(

        mean_squared_error(

            y_true,

            y_pred,

        )

    )


SCORING = {

    "MAE":

        make_scorer(

            mean_absolute_error,

            greater_is_better=False,

        ),

    "RMSE":

        make_scorer(

            rmse,

            greater_is_better=False,

        ),

    "R2":

        make_scorer(

            r2_score,

        ),

}


# ==========================================================
# Evaluate One Model
# ==========================================================

def evaluate_model(

    model,

    X,

    y,

):

    pipeline = Pipeline(

        steps=[

            (

                "preprocessor",

                create_preprocessor(

                    X.columns.tolist()

                ),

            ),

            (

                "model",

                model,

            ),

        ]

    )

    cv = create_cv()

    scores = cross_validate(

        pipeline,

        X,

        y,

        cv=cv,

        scoring=SCORING,

        n_jobs=-1,

        return_train_score=False,

    )

    results = {

        "MAE_Mean":

            -scores["test_MAE"].mean(),

        "MAE_STD":

            scores["test_MAE"].std(),

        "RMSE_Mean":

            -scores["test_RMSE"].mean(),

        "RMSE_STD":

            scores["test_RMSE"].std(),

        "R2_Mean":

            scores["test_R2"].mean(),

        "R2_STD":

            scores["test_R2"].std(),

    }

    return results

# ==========================================================
# Evaluate All Models
# ==========================================================

def evaluate_all_models(
    X,
    y,
):
    """
    Evaluate every regression model using repeated
    cross-validation.
    """

    models = create_models()

    rows = []

    print("=" * 70)
    print("Regression Model Evaluation")
    print("=" * 70)

    for name, model in models.items():

        print(f"Evaluating {name} ...")

        scores = evaluate_model(
            model,
            X,
            y,
        )

        row = {
            "Model": name,
            **scores,
        }

        rows.append(row)

    results = pd.DataFrame(rows)

    results = results.sort_values(
        by="RMSE_Mean",
        ascending=True,
    )

    results.reset_index(
        drop=True,
        inplace=True,
    )

    return results


# ==========================================================
# Save Results
# ==========================================================

def save_results(
    results,
):

    filename = (
        OUTPUT_DIR /
        "gestational_age_regression.csv"
    )

    results.to_csv(
        filename,
        index=False,
    )

    print(f"\nResults saved to:\n{filename}")


# ==========================================================
# Print Results
# ==========================================================

def print_results(
    results,
):

    print("\n")

    print("=" * 80)
    print("Regression Results")
    print("=" * 80)

    print(results.round(4))

    print("=" * 80)


# ==========================================================
# Best Model
# ==========================================================

def get_best_model(
    results,
):

    best_name = results.iloc[0]["Model"]

    print(f"\nBest model: {best_name}")

    return create_models()[best_name]


# ==========================================================
# Train Best Model
# ==========================================================

def train_best_model(
    model,
    X,
    y,
):

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                create_preprocessor(
                    X.columns.tolist()
                ),
            ),

            (
                "model",
                model,
            ),

        ]

    )

    pipeline.fit(
        X,
        y,
    )

    return pipeline


# ==========================================================
# Save Best Model
# ==========================================================

def save_best_model(
    model,
):

    filename = (
        MODEL_DIR /
        "best_gestational_age_model.pkl"
    )

    joblib.dump(
        model,
        filename,
    )

    print(f"\nBest model saved to:\n{filename}")


# ==========================================================
# Feature Summary
# ==========================================================

def save_feature_summary(
    X,
):

    summary = pd.DataFrame({

        "Feature": X.columns,

        "Missing":

            X.isna().sum().values,

        "Mean":

            X.mean().values,

        "Std":

            X.std().values,

    })

    filename = (
        OUTPUT_DIR /
        "gestational_age_features.csv"
    )

    summary.to_csv(
        filename,
        index=False,
    )

    print(f"Feature summary saved to:\n{filename}")


# ==========================================================
# Regression Summary
# ==========================================================

def save_summary(
    results,
):

    filename = (
        OUTPUT_DIR /
        "gestational_age_summary.txt"
    )

    best = results.iloc[0]

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Gestational Age Regression\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Best model : {best['Model']}\n\n"
        )

        f.write(
            f"MAE  : {best['MAE_Mean']:.4f}\n"
        )

        f.write(
            f"RMSE : {best['RMSE_Mean']:.4f}\n"
        )

        f.write(
            f"R²   : {best['R2_Mean']:.4f}\n"
        )

    print(f"Summary saved to:\n{filename}")
    
# ==========================================================
# Visualization
# ==========================================================

import matplotlib.pyplot as plt


# ==========================================================
# Prediction Plot
# ==========================================================

def save_prediction_plot(
    model,
    X,
    y,
):

    y_pred = model.predict(X)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y,
        y_pred,
        alpha=0.7,
    )

    minimum = min(
        y.min(),
        y_pred.min(),
    )

    maximum = max(
        y.max(),
        y_pred.max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        linewidth=2,
    )

    plt.xlabel(
        "Actual Gestational Age (weeks)"
    )

    plt.ylabel(
        "Predicted Gestational Age (weeks)"
    )

    plt.title(
        "Predicted vs Actual"
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR /
        "prediction_vs_actual.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {filename}")


# ==========================================================
# Residual Plot
# ==========================================================

def save_residual_plot(
    model,
    X,
    y,
):

    y_pred = model.predict(X)

    residuals = y - y_pred

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.7,
    )

    plt.axhline(
        0,
        color="red",
        linestyle="--",
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Plot"
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR /
        "residual_plot.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {filename}")


# ==========================================================
# Residual Histogram
# ==========================================================

def save_residual_histogram(
    model,
    X,
    y,
):

    residuals = y - model.predict(X)

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=20,
    )

    plt.xlabel(
        "Residual"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Residual Distribution"
    )

    plt.tight_layout()

    filename = (
        OUTPUT_DIR /
        "residual_histogram.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {filename}")


# ==========================================================
# Markdown Report
# ==========================================================

def save_markdown_report(
    results,
):

    filename = (
        OUTPUT_DIR /
        "gestational_age_report.md"
    )

    best = results.iloc[0]

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Gestational Age Regression\n\n"
        )

        f.write(
            "## Best Model\n\n"
        )

        f.write(
            f"- **Model:** {best['Model']}\n"
        )

        f.write(
            f"- **MAE:** {best['MAE_Mean']:.3f}\n"
        )

        f.write(
            f"- **RMSE:** {best['RMSE_Mean']:.3f}\n"
        )

        f.write(
            f"- **R²:** {best['R2_Mean']:.3f}\n\n"
        )

        f.write(
            "## Generated Files\n\n"
        )

        files = [

            "gestational_age_regression.csv",

            "prediction_vs_actual.png",

            "residual_plot.png",

            "residual_histogram.png",

            "best_gestational_age_model.pkl",

        ]

        for file in files:

            f.write(f"- {file}\n")

    print(f"Saved: {filename}")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("GESTATIONAL AGE REGRESSION")
    print("=" * 70)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    X, y = load_data()

    print(f"Samples : {len(X)}")

    print(f"Features: {X.shape[1]}")

    # ------------------------------------------------------
    # Evaluate models
    # ------------------------------------------------------

    results = evaluate_all_models(
        X,
        y,
    )

    print_results(
        results,
    )

    save_results(
        results,
    )

    save_feature_summary(
        X,
    )

    save_summary(
        results,
    )

    # ------------------------------------------------------
    # Train best model
    # ------------------------------------------------------

    best_model = get_best_model(
        results,
    )

    pipeline = train_best_model(
        best_model,
        X,
        y,
    )

    save_best_model(
        pipeline,
    )

    # ------------------------------------------------------
    # Plots
    # ------------------------------------------------------

    save_prediction_plot(
        pipeline,
        X,
        y,
    )

    save_residual_plot(
        pipeline,
        X,
        y,
    )

    save_residual_histogram(
        pipeline,
        X,
        y,
    )

    save_markdown_report(
        results,
    )

    print("\nFinished.")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()    