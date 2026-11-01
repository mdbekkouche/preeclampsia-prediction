from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.base import BaseEstimator
from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config
from src.models.evaluateFeatureSets import get_models   # reuse your models

class PlGFDopplerDiscordance(BaseEstimator):
    """
    Calculate the PlGF-Doppler Discordance score:

        PlGF_Doppler_Discordance = z(Mean PI) - z(PlGF)

    The mean and standard deviation for each variable are estimated
    using the training data only within each cross-validation fold.
    """

    def fit(self, X, y=None):

        features = [
            "Mean PI",
            "S-PLGF [µg/L]"
        ]

        missing = [
            col for col in features
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing demanded features: {missing}"
            )

        # Parameters estimated from the training fold only
        self.mean_pi_mean_ = X["Mean PI"].mean()
        self.mean_pi_std_ = X["Mean PI"].std(ddof=0)

        self.plgf_mean_ = X["S-PLGF [µg/L]"].mean()
        self.plgf_std_ = X["S-PLGF [µg/L]"].std(ddof=0)

        # Avoid division by zero
        if pd.isna(self.mean_pi_std_) or self.mean_pi_std_ == 0:
            self.mean_pi_std_ = 1.0

        if pd.isna(self.plgf_std_) or self.plgf_std_ == 0:
            self.plgf_std_ = 1.0

        return self

    def transform(self, X):

        X_out = X.copy()

        # Calculate z(Mean PI)
        z_mean_pi = (
            X_out["Mean PI"] - self.mean_pi_mean_
        ) / self.mean_pi_std_

        # Calculate z(PlGF)
        z_plgf = (
            X_out["S-PLGF [µg/L]"] - self.plgf_mean_
        ) / self.plgf_std_

        # Add individual z-scores
        X_out["z_Mean_PI"] = z_mean_pi
        X_out["z_S_PLGF"] = z_plgf

        # PlGF-Doppler Discordance
        X_out["PlGF_Doppler_Discordance"] = (
            z_mean_pi - z_plgf
        )

        return X_out

PLACENTAL_FEATURES = [
    "Mean PI",
    "S-Flt1 [µg/L]",
    "S-PLGF [µg/L]"
]

# ============================================================
# Placental Dysfunction Score
# ============================================================

class PlacentalDysfunctionScore(BaseEstimator):
    """
    Calculate:
        PDS = z(Mean PI) + z(sFlt-1) - z(PlGF)

    Means and standard deviations are learned only from the
    training portion of each cross-validation fold.
    """

    def fit(self, X, y=None):
        missing = [
            col for col in PLACENTAL_FEATURES
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required biomarker columns: {missing}"
            )

        self.means_ = X[PLACENTAL_FEATURES].mean()
        self.stds_ = X[PLACENTAL_FEATURES].std(ddof=0)

        # Prevent division by zero for constant or entirely
        # missing features. Missing values are imputed later.
        self.stds_ = self.stds_.replace(0, 1).fillna(1)

        return self

    def transform(self, X):
        X_out = X.copy()

        z_mean_pi = (
            X_out["Mean PI"] - self.means_["Mean PI"]
        ) / self.stds_["Mean PI"]

        z_sflt1 = (
            X_out["S-Flt1 [µg/L]"]
            - self.means_["S-Flt1 [µg/L]"]
        ) / self.stds_["S-Flt1 [µg/L]"]

        z_plgf = (
            X_out["S-PLGF [µg/L]"]
            - self.means_["S-PLGF [µg/L]"]
        ) / self.stds_["S-PLGF [µg/L]"]

        X_out["z_Mean_PI"] = z_mean_pi
        X_out["z_S_Flt1"] = z_sflt1
        X_out["z_S_PLGF"] = z_plgf

        X_out["Placental_Dysfunction_Score"] = (
            z_mean_pi + z_sflt1 - z_plgf
        )

        return X_out



def evaluate_selective_prediction(model, X, y, cv, threshold):

    y_true = []
    y_pred = []
    y_prob = []

    total_samples = 0
    accepted_samples = 0

    for train_idx, test_idx in cv.split(X, y):

        clf = clone(model)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)

        confidence = np.max(proba, axis=1)

        keep = confidence >= threshold

        total_samples += len(y_test)
        accepted_samples += keep.sum()

        if keep.sum() == 0:
            continue

        y_true.extend(y_test[keep])

        y_pred.extend(np.argmax(proba[keep], axis=1))

        y_prob.extend(proba[keep][:, 1])

    coverage = accepted_samples / total_samples

    if len(np.unique(y_true)) < 2:
        return None

    return {
        "Coverage": coverage,
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }
    
def main():

    config = load_config("config/config.yaml")

    df = load_dataset(config["data"]["raw_file"])
    df = df.drop(columns=["Gestational age at delivery [weeks]"])
    df = df.drop(columns=["Birth weight [g]"])

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

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    models = get_models()

    results = []

    for model_name, model in models.items():

        print(model_name)

        if not hasattr(model, "predict_proba"):
            continue

        for threshold in thresholds:
            
            pipeline = Pipeline([
                ("plgf_doppler_discordance", PlGFDopplerDiscordance()),
                ("placental_score", PlacentalDysfunctionScore()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ])
            
            metrics = evaluate_selective_prediction(
                pipeline,
                X,
                y,
                cv,
                threshold,
            )

            if metrics is None:
                continue

            metrics["Model"] = model_name
            metrics["Threshold"] = threshold

            results.append(metrics)

    results = pd.DataFrame(results)

    Path("results").mkdir(exist_ok=True)

    results.to_csv(
        "results/selective_prediction.csv",
        index=False,
    )

    print(results)    
    
if __name__ == "__main__":
    main()    