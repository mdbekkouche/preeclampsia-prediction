"""
Global SHAP analysis for the Random Forest preeclampsia classifier
using repeated stratified 5-fold cross-validation (5 folds x 10 repeats).

For each outer fold:
  1. Fit the complete preprocessing + Random Forest pipeline on the
     training fold only.
  2. Fit a fold-specific SHAP background using training-fold samples.
  3. Calculate positive-class SHAP values only for the held-out fold.

Global feature importance is calculated by averaging absolute SHAP
contributions across the 10 held-out explanations for each patient,
then averaging across patients.

Since every patient is held out exactly once per repetition, this is
equivalent to averaging |SHAP| over all held-out evaluation instances.

Outputs (results/shap_random_forest_cv/):
  - feature_importance.csv
  - shap_contributions.csv
  - shap_global_bar.png
  - shap_global_beeswarm.png
  - shap_global_explanation.pkl

Expected project structure:
  - config/config.yaml
  - src/data/load_data.py
  - src/utils/config.py
  - results/models/best_model.pkl (must contain a fitted RandomForestClassifier
    or a Pipeline whose final step is the Random Forest classifier)
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_dataset
from src.utils.config import load_config


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CONFIG_FILE = "config/config.yaml"
MODEL_FILE = "results/models/best_model.pkl"
OUTPUT_DIR = Path("results/shap_random_forest_cv")

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42
BACKGROUND_SIZE = 30
MAX_DISPLAY = 20
EPSILON = 1e-10

# These two features are estimated from training-fold statistics and
# must therefore be generated inside the cross-validation pipeline.
FOLD_DERIVED_FEATURES = {
    "Placental_Dysfunction_Score",
    "PlGF_Doppler_Discordance",
}

# Biomarker and Doppler column names used by the uploaded project code.
MEAN_PI_COLUMN = "Mean PI"
SFLT1_COLUMN = "S-Flt1 [µg/L]"
PLGF_COLUMN = "S-PLGF [µg/L]"

warnings.filterwarnings("default", category=UserWarning)


# ---------------------------------------------------------------------
# Fold-fitted feature calculations
# ---------------------------------------------------------------------

class PlGFDopplerDiscordance(BaseEstimator, TransformerMixin):
    """Add PlGF_Doppler_Discordance = z(Mean PI) - z(PlGF).

    Mean and standard deviation are fitted on the current training fold.
    The fitted values are then reused unchanged for validation samples.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self._check_input(X)
        self.mean_pi_mean_ = float(X[MEAN_PI_COLUMN].mean())
        self.mean_pi_std_ = float(X[MEAN_PI_COLUMN].std(ddof=0))
        self.plgf_mean_ = float(X[PLGF_COLUMN].mean())
        self.plgf_std_ = float(X[PLGF_COLUMN].std(ddof=0))

        if not np.isfinite(self.mean_pi_mean_) or not np.isfinite(self.plgf_mean_):
            raise ValueError("Mean PI or PlGF has no observed values in this training fold.")
        self.mean_pi_std_ = self._safe_std(self.mean_pi_std_)
        self.plgf_std_ = self._safe_std(self.plgf_std_)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @staticmethod
    def _safe_std(value: float) -> float:
        if not np.isfinite(value) or value == 0:
            return 1.0
        return value

    @staticmethod
    def _check_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PlGFDopplerDiscordance expects a pandas DataFrame.")
        missing = [c for c in (MEAN_PI_COLUMN, PLGF_COLUMN) if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for PlGF-Doppler discordance: {missing}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_input(X)
        X_out = X.copy()
        z_mean_pi = (X_out[MEAN_PI_COLUMN] - self.mean_pi_mean_) / self.mean_pi_std_
        z_plgf = (X_out[PLGF_COLUMN] - self.plgf_mean_) / self.plgf_std_
        X_out["PlGF_Doppler_Discordance"] = z_mean_pi - z_plgf
        return X_out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        if "PlGF_Doppler_Discordance" not in names:
            names.append("PlGF_Doppler_Discordance")
        return np.asarray(names, dtype=object)


class PlacentalDysfunctionScore(BaseEstimator, TransformerMixin):
    """Add PDS = z(Mean PI) + z(sFlt-1) - z(PlGF).

    Mean and standard deviation are fitted on the current training fold.
    The fitted values are then reused unchanged for validation samples.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self._check_input(X)
        self.mean_pi_mean_ = float(X[MEAN_PI_COLUMN].mean())
        self.mean_pi_std_ = float(X[MEAN_PI_COLUMN].std(ddof=0))
        self.sflt1_mean_ = float(X[SFLT1_COLUMN].mean())
        self.sflt1_std_ = float(X[SFLT1_COLUMN].std(ddof=0))
        self.plgf_mean_ = float(X[PLGF_COLUMN].mean())
        self.plgf_std_ = float(X[PLGF_COLUMN].std(ddof=0))

        if not all(np.isfinite(v) for v in (
            self.mean_pi_mean_, self.sflt1_mean_, self.plgf_mean_
        )):
            raise ValueError(
                "Mean PI, sFlt-1, or PlGF has no observed values in this training fold."
            )
        self.mean_pi_std_ = self._safe_std(self.mean_pi_std_)
        self.sflt1_std_ = self._safe_std(self.sflt1_std_)
        self.plgf_std_ = self._safe_std(self.plgf_std_)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @staticmethod
    def _safe_std(value: float) -> float:
        if not np.isfinite(value) or value == 0:
            return 1.0
        return value

    @staticmethod
    def _check_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PlacentalDysfunctionScore expects a pandas DataFrame.")
        missing = [
            c for c in (MEAN_PI_COLUMN, SFLT1_COLUMN, PLGF_COLUMN)
            if c not in X.columns
        ]
        if missing:
            raise ValueError(f"Missing columns for placental dysfunction score: {missing}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_input(X)
        X_out = X.copy()
        z_mean_pi = (X_out[MEAN_PI_COLUMN] - self.mean_pi_mean_) / self.mean_pi_std_
        z_sflt1 = (X_out[SFLT1_COLUMN] - self.sflt1_mean_) / self.sflt1_std_
        z_plgf = (X_out[PLGF_COLUMN] - self.plgf_mean_) / self.plgf_std_
        X_out["Placental_Dysfunction_Score"] = z_mean_pi + z_sflt1 - z_plgf
        return X_out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        if "Placental_Dysfunction_Score" not in names:
            names.append("Placental_Dysfunction_Score")
        return np.asarray(names, dtype=object)


# ---------------------------------------------------------------------
# Data loading and leakage-safe feature preparation
# ---------------------------------------------------------------------

def load_data():
    """Load source data and create only row-wise deterministic features.

    The two z-score-derived features are deliberately excluded from X
    here and are created by the fold-fitted classes above. This prevents
    use of full-dataset means or standard deviations before CV splitting.
    """
    config = load_config(CONFIG_FILE)
    df = load_dataset(config["data"]["raw_file"]).copy()

    df = df.drop(
        columns=[
            "Gestational age at delivery [weeks]",
            "Birth weight [g]",
        ],
        errors="ignore",
    )

    target = config["target"]["column"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    feature_config = config["features"]
    configured_features = (
        feature_config.get("clinical", [])
        + feature_config.get("doppler", [])
        + feature_config.get("biomarkers", [])
        + feature_config.get("interaction_features", [])
    )
    

    # Construct only deterministic, per-patient interaction features.
    # Any feature whose parameters depend on the cohort (the z-score-based
    # scores) is generated later inside each training fold.
    required = [MEAN_PI_COLUMN, SFLT1_COLUMN, PLGF_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns needed to build interaction features: {missing}")

    sflt1 = pd.to_numeric(df[SFLT1_COLUMN], errors="coerce")
    plgf = pd.to_numeric(df[PLGF_COLUMN], errors="coerce")
    mean_pi = pd.to_numeric(df[MEAN_PI_COLUMN], errors="coerce")

    ratio = sflt1 / (plgf + EPSILON)
    df["log_sFlt1_PlGF"] = np.log((sflt1 + EPSILON) / (plgf + EPSILON))
    df["Doppler_Angiogenic_Index"] = mean_pi * df["log_sFlt1_PlGF"]

    # Some project configurations name the raw ratio differently. Create
    # the ratio only under names actually requested by the configuration.
    ratio_feature_names = [
        name for name in configured_features
        if ("sflt" in name.lower() and "plgf" in name.lower()
            and ("ratio" in name.lower() or "/" in name))
    ]
    for ratio_name in ratio_feature_names:
        df[ratio_name] = ratio

    # Remove any precomputed full-dataset versions of the two z-score
    # features: the pipeline will create their fold-specific versions.
    configured_features = [
        f for f in dict.fromkeys(configured_features)
        if f not in FOLD_DERIVED_FEATURES
    ]

    features = [f for f in configured_features if f in df.columns]
    missing_features = [f for f in configured_features if f not in df.columns]
    if missing_features:
        print("Warning: configured features not present and omitted:", missing_features)

    if not features:
        raise ValueError("No configured predictor columns were found in the dataset.")

    df = df.dropna(subset=[target]).copy()
    X = df[features].copy()

    # The supplied pipeline uses median imputation, so ensure predictors
    # are numerical and represent infinities as missing values.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            "These predictors contain no observed values and cannot be imputed: "
            + ", ".join(all_missing)
        )

    y_raw = df.loc[X.index, target]
    y = y_raw.isin(["PE", "IUGR+PE"]).astype(int)

    return X, y


# ---------------------------------------------------------------------
# Load the Random Forest estimator
# ---------------------------------------------------------------------

def load_random_forest_model():
    """Load Random Forest from the configured model artifact.

    The artifact may contain either a RandomForestClassifier directly
    or a Pipeline whose final step is the classifier.
    """
    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        raise FileNotFoundError(f"Random Forest model artifact not found: {model_path}")

    saved = joblib.load(model_path)

    # Support a saved GridSearchCV object.
    if hasattr(saved, "best_estimator_"):
        saved = saved.best_estimator_

    # If the saved artifact is a pipeline, extract its final estimator.
    if isinstance(saved, Pipeline):
        if "model" in saved.named_steps:
            saved = saved.named_steps["model"]
        else:
            saved = saved.steps[-1][1]

    if not isinstance(saved, RandomForestClassifier):
        raise TypeError(
            f"Expected RandomForestClassifier in {MODEL_FILE}, but found "
            f"{saved.__class__.__name__}. Set MODEL_FILE to the saved Random Forest artifact."
        )

    print(f"Loaded Random Forest estimator from: {MODEL_FILE}")
    return clone(saved)


# ---------------------------------------------------------------------
# Create complete pipeline
# ---------------------------------------------------------------------

def make_pipeline(model):
    """Build the fold-fitted pipeline specified for this analysis."""
    return Pipeline([
        ("plgf_doppler_discordance", PlGFDopplerDiscordance()),
        ("placental_score", PlacentalDysfunctionScore()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clone(model)),
    ])


def transform_before_model(fitted_pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted pipeline steps preceding the Random Forest.

    Return a DataFrame so SHAP retains the names of the original and
    engineered model inputs.
    """
    X_step = X.copy()
    X_step = fitted_pipeline.named_steps["plgf_doppler_discordance"].transform(X_step)
    X_step = fitted_pipeline.named_steps["placental_score"].transform(X_step)
    feature_names = list(X_step.columns)

    X_imputed = fitted_pipeline.named_steps["imputer"].transform(X_step)
    if X_imputed.shape[1] != len(feature_names):
        raise ValueError(
            "SimpleImputer changed the number of features. Check for columns "
            "that are entirely missing in a training fold."
        )

    X_scaled = fitted_pipeline.named_steps["scaler"].transform(X_imputed)
    return pd.DataFrame(X_scaled, columns=feature_names, index=X.index)


# ---------------------------------------------------------------------
# SHAP output normalization
# ---------------------------------------------------------------------

def explain_positive_class(explainer, X_eval: pd.DataFrame, positive_class_index: int):
    """Return positive-class SHAP values and base values across SHAP versions."""
    try:
        explanation = explainer(X_eval, check_additivity=False)
    except TypeError:
        explanation = explainer(X_eval)

    values = explanation.values
    base_values = explanation.base_values

    # Compatibility with older SHAP versions that return a list by class.
    if isinstance(values, list):
        values = values[positive_class_index]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, positive_class_index]

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    base_values = np.asarray(base_values)
    if base_values.ndim == 2:
        base_values = base_values[:, positive_class_index]
    elif base_values.ndim == 1 and base_values.size > 1:
        # A one-dimensional class vector (rather than one value per row).
        if base_values.size != len(X_eval):
            base_values = np.repeat(base_values[positive_class_index], len(X_eval))
    elif base_values.ndim == 0:
        base_values = np.repeat(float(base_values), len(X_eval))

    base_values = np.asarray(base_values).reshape(-1)
    if base_values.size == 1:
        base_values = np.repeat(base_values[0], len(X_eval))
    if base_values.size != len(X_eval):
        raise ValueError(f"Unexpected SHAP base-value shape: {base_values.shape}")

    return values, base_values


# ---------------------------------------------------------------------
# Global SHAP calculation and output
# ---------------------------------------------------------------------

def run_cross_validated_shap(X: pd.DataFrame, y: pd.Series, base_model):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = y.value_counts()
    if class_counts.min() < N_SPLITS:
        raise ValueError(
            f"Each class must contain at least {N_SPLITS} samples; "
            f"observed class counts are {class_counts.to_dict()}"
        )

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # Store all held-out SHAP explanations, with one explanation per
    # patient in each repetition.
    shap_values_parts = []
    shap_data_parts = []
    base_value_parts = []
    oof_records = []

    # Patient-level accumulation of absolute SHAP values. This allows
    # each patient's R repeated held-out explanations to be averaged
    # before calculating the final across-patient mean.
    patient_abs_sum = None
    patient_counts = np.zeros(len(X), dtype=int)
    feature_names_reference = None

    total_folds = N_SPLITS * N_REPEATS

    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        repetition = (fold_number - 1) // N_SPLITS + 1
        fold_in_repetition = (fold_number - 1) % N_SPLITS + 1

        print(
            f"Fold {fold_number:02d}/{total_folds} "
            f"(repetition {repetition}, fold {fold_in_repetition})"
        )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # Fit all preprocessing and the model only on the training fold.
        fold_pipeline = make_pipeline(base_model)
        fold_pipeline.fit(X_train, y_train)

        rf = fold_pipeline.named_steps["model"]
        classes = list(rf.classes_)
        if 1 not in classes:
            raise ValueError(f"Positive class label 1 is absent from fold {fold_number}.")
        positive_class_index = classes.index(1)

        # Transform training and validation data with fold-fitted steps.
        X_train_transformed = transform_before_model(fold_pipeline, X_train)
        X_valid_transformed = transform_before_model(fold_pipeline, X_valid)

        feature_names = list(X_valid_transformed.columns)
        if feature_names_reference is None:
            feature_names_reference = feature_names
            patient_abs_sum = np.zeros((len(X), len(feature_names)), dtype=float)
        elif feature_names != feature_names_reference:
            raise ValueError("Feature columns or their order changed between CV folds.")

        # Training-fold background only: no validation observations enter
        # the SHAP reference dataset.
        n_background = min(BACKGROUND_SIZE, len(X_train_transformed))
        background = X_train_transformed.sample(
            n=n_background,
            random_state=RANDOM_STATE + fold_number,
        )

        explainer = shap.TreeExplainer(
            rf,
            data=background,
            feature_perturbation="interventional",
            model_output="probability",
        )

        fold_values, fold_base_values = explain_positive_class(
            explainer,
            X_valid_transformed,
            positive_class_index,
        )

        if fold_values.shape[1] != len(feature_names):
            raise ValueError(
                f"SHAP returned {fold_values.shape[1]} features, "
                f"but the pipeline produced {len(feature_names)}."
            )

        # Collect all out-of-fold explanations.
        shap_values_parts.append(fold_values)
        shap_data_parts.append(X_valid_transformed.to_numpy())
        base_value_parts.append(fold_base_values)

        # Accumulate mean absolute contribution for each original patient.
        patient_abs_sum[valid_idx] += np.abs(fold_values)
        patient_counts[valid_idx] += 1

        # Save held-out predictions and per-feature SHAP contributions.
        valid_probabilities = fold_pipeline.predict_proba(X_valid)[:, positive_class_index]
        for local_row, original_position in enumerate(valid_idx):
            record = {
                "patient_position": int(original_position),
                "source_index": str(X.index[original_position]),
                "repetition": repetition,
                "fold": fold_in_repetition,
                "y_true": int(y_valid.iloc[local_row]),
                "predicted_probability": float(valid_probabilities[local_row]),
                "base_value": float(fold_base_values[local_row]),
            }
            for feature_idx, feature_name in enumerate(feature_names):
                record[f"SHAP::{feature_name}"] = float(fold_values[local_row, feature_idx])
            oof_records.append(record)

    # Every patient must be held out exactly once in each repetition.
    if not np.all(patient_counts == N_REPEATS):
        raise RuntimeError(
            "Unexpected number of held-out explanations per patient. "
            f"Observed counts: {np.unique(patient_counts)}"
        )

    all_shap_values = np.vstack(shap_values_parts)
    all_shap_data = np.vstack(shap_data_parts)
    all_base_values = np.concatenate(base_value_parts)

    # Average |SHAP| across repetitions for each patient, then across
    # the n unique patients. This implements the requested I_j formula.
    patient_mean_abs = patient_abs_sum / patient_counts[:, np.newaxis]
    global_importance = patient_mean_abs.mean(axis=0)

    importance_df = (
        pd.DataFrame({
            "Feature": feature_names_reference,
            "MeanAbsSHAP": global_importance,
        })
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )

    importance_path = OUTPUT_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    oof_path = OUTPUT_DIR / "shap_contributions.csv"
    pd.DataFrame(oof_records).to_csv(oof_path, index=False)

    # Build a combined explanation object for global plotting. It contains
    # all held-out explanations (n patients x n repetitions).
    global_explanation = shap.Explanation(
        values=all_shap_values,
        base_values=all_base_values,
        data=all_shap_data,
        feature_names=feature_names_reference,
    )
    joblib.dump(global_explanation, OUTPUT_DIR / "shap_global_explanation.pkl")

    # Global bar plot based on all held-out explanations.
    plt.figure(figsize=(9, 7))
    shap.plots.bar(
        global_explanation,
        max_display=min(MAX_DISPLAY, len(feature_names_reference)),
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Global beeswarm plot showing direction and distribution of SHAP values.
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(
        global_explanation,
        max_display=min(MAX_DISPLAY, len(feature_names_reference)),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_global_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\n" + "=" * 72)
    print("GLOBAL SHAP FEATURE IMPORTANCE — RANDOM FOREST")
    print("=" * 72)
    print(f"Unique patients                 : {len(X)}")
    print(f"Cross-validation evaluations    : {total_folds}")
    print(f"Held-out explanations           : {len(all_shap_values)}")
    print(f"Held-out explanations per patient: {N_REPEATS}")
    print("\nImportance: mean absolute SHAP contribution")
    print(importance_df.to_string(index=False, formatters={"MeanAbsSHAP": "{:.6f}".format}))
    print("\nSaved outputs:")
    print(f"- {importance_path}")
    print(f"- {oof_path}")
    print(f"- {OUTPUT_DIR / 'shap_global_bar.png'}")
    print(f"- {OUTPUT_DIR / 'shap_global_beeswarm.png'}")
    print(f"- {OUTPUT_DIR / 'shap_global_explanation.pkl'}")
    print("=" * 72)

    return importance_df, global_explanation


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Loading preeclampsia dataset...")
    X, y = load_data()

    print(f"Samples : {len(X)}")
    print(f"Features: {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts().sort_index().rename(index={0: "Control", 1: "Preeclampsia"}))

    print("\nLoading Random Forest estimator...")
    random_forest = load_random_forest_model()

    print("\nRunning repeated cross-validated SHAP analysis...")
    run_cross_validated_shap(X, y, random_forest)


if __name__ == "__main__":
    main()
