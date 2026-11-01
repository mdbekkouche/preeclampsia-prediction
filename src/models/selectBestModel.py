from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config
from src.models.evaluateFeatureSets import get_models

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

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
# 2. Placental Dysfunction Score
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


def main():

    config = load_config("config/config.yaml")

    df = load_dataset(config["data"]["raw_file"])

    df = build_interaction_features(df, config)

    target = config["target"]["column"]
    
   
    features = (
        config["features"]["clinical"]
        + config["features"]["doppler"]
        + config["features"]["biomarkers"]
        + config["features"]["interaction_features"]
    )
  
    X = df.drop(columns=[target])

    
    features = [f for f in features if f in df.columns]
    '''
    X = df[features]
    '''
    
    y = df[target]
    
    y = y.isin(
            ["PE", "IUGR+PE"]
        ).astype(int)
    
    models = get_models()

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, average="weighted")
    }

    results = []

    best_model = None
    best_name = None
    best_auc = -1
    best_f1 = -1

    print("\nEvaluating models...\n")

    for name, model in models.items():

        pipeline = Pipeline([
            ("plgf_doppler_discordance", PlGFDopplerDiscordance()),
            ("placental_score", PlacentalDysfunctionScore()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
            
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise"
        )

        auc = scores["test_roc_auc"].mean()
        f1 = scores["test_f1"].mean()

        results.append({
            "Model": name,
            "ROC_AUC": auc,
            "F1": f1
        })

        print(
            f"{name:25s}"
            f"ROC-AUC={auc:.4f} "
            f"F1={f1:.4f}"
        )

        if (
            auc > best_auc or
            (abs(auc - best_auc) < 1e-6 and f1 > best_f1)
        ):

            best_auc = auc
            best_f1 = f1
            best_model = model
            best_name = name

    print("\nBest model:", best_name)

    print("Retraining on the complete dataset...")

    best_model.fit(X, y)

    output_dir = Path("results/models")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        output_dir / "best_model.pkl"
    )

    pd.DataFrame(results).sort_values(
        "ROC_AUC",
        ascending=False
    ).to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    info = pd.DataFrame([{
        "Model": best_name,
        "ROC_AUC": best_auc,
        "F1": best_f1,
        "Num_Features": len(features)
    }])

    info.to_csv(
        output_dir / "best_model_info.csv",
        index=False
    )

    with open(
        output_dir / "selected_features.json",
        "w"
    ) as f:

        json.dump(
            features,
            f,
            indent=4
        )

    print("\nModel saved successfully.")

    print(output_dir / "best_model.pkl")


if __name__ == "__main__":
    main()