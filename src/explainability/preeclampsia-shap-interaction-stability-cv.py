"""
Pairwise SHAP interaction stability analysis for the Random Forest
preeclampsia classifier using repeated stratified 5-fold CV (10 repeats).

For each outer fold:
  1. Fit the full preprocessing + Random Forest pipeline on training data.
  2. Calculate pairwise SHAP interaction values for the held-out fold.
  3. Exclude diagonal (main-effect) entries and rank off-diagonal pairs.
  4. Aggregate pair strength and rank statistics across all 50 folds.

Interaction values use TreeExplainer with tree_path_dependent perturbation
and raw model output, the supported TreeSHAP configuration for interaction
values. For sklearn RandomForestClassifier, SHAP returns class-specific
outputs; the positive (preeclampsia) class is selected.

Outputs (results/shap_interaction_stability/):
  - interaction_stability_summary.csv
  - fold_pair_interaction_ranking.csv
  - engineered_feature_interactions.csv
  - mean_interaction_matrix.csv
  - top_interactions_bar.png

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
OUTPUT_DIR = Path("results/shap_interaction_stability")

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
# Create the requested complete pipeline
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
# Pairwise SHAP interaction utilities
# ---------------------------------------------------------------------

TOP_N_PLOT = 20
TOP_K_FREQUENCY = (5, 10)
ENGINEERED_FEATURES = {
    "Placental_Dysfunction_Score",
    "PlGF_Doppler_Discordance",
    "log_sFlt1_PlGF",
    "Doppler_Angiogenic_Index",
}


def normalize_interaction_values(raw_values, positive_class_index, n_samples, n_features):
    """Normalize SHAP interaction output to (samples, features, features).

    SHAP versions before 0.45 may return a list containing one tensor per
    class. Newer versions return an ndarray with the output/class dimension
    at the last axis.
    """
    if isinstance(raw_values, list):
        if positive_class_index >= len(raw_values):
            raise ValueError("Positive-class index is outside SHAP output list.")
        values = np.asarray(raw_values[positive_class_index])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 4:
            # Current SHAP convention: (samples, features, features, outputs).
            if values.shape[:3] == (n_samples, n_features, n_features):
                if positive_class_index >= values.shape[3]:
                    raise ValueError("Positive-class index is outside SHAP output axis.")
                values = values[:, :, :, positive_class_index]
            # Defensive compatibility for output-first tensors.
            elif values.shape[1:] == (n_samples, n_features, n_features):
                if positive_class_index >= values.shape[0]:
                    raise ValueError("Positive-class index is outside SHAP output axis.")
                values = values[positive_class_index]
            else:
                raise ValueError(f"Unrecognized 4-D SHAP interaction shape: {values.shape}")

    if values.shape != (n_samples, n_features, n_features):
        raise ValueError(
            "Expected SHAP interaction array with shape "
            f"({n_samples}, {n_features}, {n_features}), got {values.shape}."
        )

    if not np.isfinite(values).all():
        raise ValueError("SHAP interaction values contain NaN or infinite values.")

    # Interaction matrices are symmetric. Symmetrize tiny numerical
    # differences to make the pairwise aggregation unambiguous.
    return 0.5 * (values + np.swapaxes(values, 1, 2))


def build_pairwise_fold_table(interaction_values, feature_names, repetition, fold_in_repetition):
    """Calculate mean absolute off-diagonal interaction strength per pair.

    The diagonal is excluded because it represents main effects, not
    pairwise interactions. Only the upper triangle is retained to avoid
    counting each symmetric pair twice.
    """
    n_features = len(feature_names)
    upper_i, upper_j = np.triu_indices(n_features, k=1)

    # Mean absolute interaction across held-out patients in this fold.
    mean_abs_matrix = np.abs(interaction_values).mean(axis=0)

    pair_strengths = mean_abs_matrix[upper_i, upper_j]
    ranks = pd.Series(pair_strengths).rank(
        ascending=False, method="min"
    ).astype(int).to_numpy()

    fold_table = pd.DataFrame({
        "Repetition": repetition,
        "Fold": fold_in_repetition,
        "Feature_1": [feature_names[i] for i in upper_i],
        "Feature_2": [feature_names[j] for j in upper_j],
        "MeanAbsInteraction": pair_strengths,
        "Rank": ranks,
        "N_HeldOut": interaction_values.shape[0],
    })
    return fold_table, mean_abs_matrix


def summarize_interaction_stability(fold_results):
    """Aggregate pair strength and rankings across outer CV folds."""
    summary = (
        fold_results
        .groupby(["Feature_1", "Feature_2"], as_index=False)
        .agg(
            MeanAbsInteraction=("MeanAbsInteraction", "mean"),
            SD_Interaction=("MeanAbsInteraction", "std"),
            MeanRank=("Rank", "mean"),
            RankSD=("Rank", "std"),
        )
    )

    for k in TOP_K_FREQUENCY:
        frequency = (
            fold_results.assign(InTopK=fold_results["Rank"] <= k)
            .groupby(["Feature_1", "Feature_2"])['InTopK']
            .mean()
            .mul(100)
            .rename(f"Top{k}Frequency_pct")
            .reset_index()
        )
        summary = summary.merge(
            frequency, on=["Feature_1", "Feature_2"], how="left"
        )

    summary["Interaction"] = (
        summary["Feature_1"].astype(str)
        + " × "
        + summary["Feature_2"].astype(str)
    )
    summary = summary.sort_values(
        ["MeanAbsInteraction", "MeanRank"],
        ascending=[False, True]
    ).reset_index(drop=True)
    summary.insert(0, "OverallRank", np.arange(1, len(summary) + 1))
    return summary


def create_interaction_matrix(summary, feature_names):
    """Build a symmetric matrix of mean absolute pair interaction strengths."""
    matrix = pd.DataFrame(
        np.zeros((len(feature_names), len(feature_names)), dtype=float),
        index=feature_names,
        columns=feature_names,
    )
    for row in summary.itertuples(index=False):
        matrix.loc[row.Feature_1, row.Feature_2] = row.MeanAbsInteraction
        matrix.loc[row.Feature_2, row.Feature_1] = row.MeanAbsInteraction
    return matrix


def save_top_interactions_plot(summary):
    """Save a vertical bar plot for the strongest SHAP interactions."""

    top = summary.head(TOP_N_PLOT).copy()

    fig_width = max(12, 0.8 * len(top))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bars = ax.bar(
        top["Interaction"],
        top["MeanAbsInteraction"],
        yerr=top["SD_Interaction"].fillna(0),
        capsize=3,
        color="#004400"
    )
    
    

    ax.set_xlabel("Feature pairs", fontsize=11)
    ax.set_ylabel("Mean absolute SHAP interaction value", fontsize=11)

    ax.set_title(
        "Top pairwise SHAP interactions across CV folds",
        fontsize=13
    )

    # Rotate feature-pair labels
    plt.xticks(
        rotation=65,
        ha="right",
        fontsize=9
    )

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "top_interactions_bar_vertical.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

def run_interaction_stability(X: pd.DataFrame, y: pd.Series, base_model):
    """Compute pairwise SHAP interaction stability across repeated CV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = y.value_counts()
    if class_counts.min() < N_SPLITS:
        raise ValueError(
            f"Each class must contain at least {N_SPLITS} samples; "
            f"observed class counts: {class_counts.to_dict()}"
        )

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    all_fold_tables = []
    feature_names_reference = None
    interaction_matrix_sum = None
    interaction_matrix_count = 0
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

        # Fit the complete requested pipeline on this training fold only.
        fold_pipeline = make_pipeline(base_model)
        fold_pipeline.fit(X_train, y_train)

        rf = fold_pipeline.named_steps["model"]
        if not isinstance(rf, RandomForestClassifier):
            raise TypeError(
                "The final pipeline step must be RandomForestClassifier; "
                f"received {rf.__class__.__name__}."
            )

        classes = list(rf.classes_)
        if 1 not in classes:
            raise ValueError(
                f"Positive class label 1 is absent from fold {fold_number}."
            )
        positive_class_index = classes.index(1)

        # Apply fold-fitted feature engineering, imputation and scaling.
        X_valid_transformed = transform_before_model(fold_pipeline, X_valid)
        feature_names = list(X_valid_transformed.columns)
        n_features = len(feature_names)

        if feature_names_reference is None:
            feature_names_reference = feature_names
            interaction_matrix_sum = np.zeros(
                (n_features, n_features), dtype=float
            )
        elif feature_names != feature_names_reference:
            raise ValueError(
                "Feature names/order changed between folds; interaction "
                "matrices cannot be aggregated safely."
            )

        # Interaction values are computed with tree_path_dependent and raw
        # output. For this mode, TreeExplainer uses the fold-trained trees'
        # path-cover statistics as its reference distribution; no validation
        # samples are used as a background dataset.
        interaction_explainer = shap.TreeExplainer(
            rf,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        raw_interactions = interaction_explainer.shap_interaction_values(
            X_valid_transformed
        )
        interaction_values = normalize_interaction_values(
            raw_interactions,
            positive_class_index=positive_class_index,
            n_samples=len(X_valid_transformed),
            n_features=n_features,
        )

        fold_table, fold_matrix = build_pairwise_fold_table(
            interaction_values,
            feature_names,
            repetition=repetition,
            fold_in_repetition=fold_in_repetition,
        )
        all_fold_tables.append(fold_table)

        interaction_matrix_sum += fold_matrix
        interaction_matrix_count += 1

        print(
            f"  Held-out patients: {len(X_valid)} | "
            f"Features: {n_features} | "
            f"Pairs: {len(fold_table)}"
        )

    # Save per-fold pair strengths and ranks (one row per pair per fold).
    fold_results = pd.concat(all_fold_tables, ignore_index=True)
    fold_results_path = OUTPUT_DIR / "fold_pair_interaction_ranking.csv"
    fold_results.to_csv(fold_results_path, index=False)

    # Aggregate strength and rank stability across the 50 outer folds.
    summary = summarize_interaction_stability(fold_results)
    summary_path = OUTPUT_DIR / "interaction_stability_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Save interactions involving any of the four engineered features.
    engineered_mask = (
        summary["Feature_1"].isin(ENGINEERED_FEATURES)
        | summary["Feature_2"].isin(ENGINEERED_FEATURES)
    )
    engineered_summary = summary.loc[engineered_mask].copy()
    engineered_path = OUTPUT_DIR / "engineered_feature_interactions.csv"
    engineered_summary.to_csv(engineered_path, index=False)

    # Mean pairwise interaction matrix across outer folds.
    mean_matrix = interaction_matrix_sum / interaction_matrix_count
    matrix_df = pd.DataFrame(
        mean_matrix,
        index=feature_names_reference,
        columns=feature_names_reference,
    )
    matrix_path = OUTPUT_DIR / "mean_interaction_matrix.csv"
    matrix_df.to_csv(matrix_path, index_label="Feature")

    save_top_interactions_plot(summary)

    print("\n" + "=" * 78)
    print("PAIRWISE SHAP INTERACTION STABILITY — RANDOM FOREST")
    print("=" * 78)
    print(f"Outer evaluations : {interaction_matrix_count}")
    print(f"Features          : {len(feature_names_reference)}")
    print(f"Unique pairs      : {len(summary)}")
    display_cols = [
        "OverallRank", "Interaction", "MeanAbsInteraction",
        "SD_Interaction", "MeanRank", "Top5Frequency_pct",
        "Top10Frequency_pct"
    ]
    print("\nTop 20 pairwise interactions:")
    print(
        summary[display_cols].head(20).to_string(
            index=False,
            formatters={
                "MeanAbsInteraction": "{:.6f}".format,
                "SD_Interaction": lambda x: "NA" if pd.isna(x) else f"{x:.6f}",
                "MeanRank": "{:.2f}".format,
                "Top5Frequency_pct": "{:.1f}".format,
                "Top10Frequency_pct": "{:.1f}".format,
            },
        )
    )

    print("\nSaved outputs:")
    for path in (
        fold_results_path,
        summary_path,
        engineered_path,
        matrix_path,
        OUTPUT_DIR / "top_interactions_bar.png",
    ):
        print(f"- {path}")
    print("=" * 78)

    return summary, engineered_summary, fold_results


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

    print("\nRunning pairwise SHAP interaction stability analysis...")
    run_interaction_stability(X, y, random_forest)


if __name__ == "__main__":
    main()
