from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config

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


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ),

        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        # KNN does not support class_weight
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),

        # GaussianNB does not support class_weight
        "NaiveBayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GaussianNB())
        ]),
    }


def main():

    config = load_config("config/config.yaml")

    df = load_dataset(config["data"]["raw_file"])
    df = df.drop(columns=["Gestational age at delivery [weeks]"])
    df = df.drop(columns=["Birth weight [g]"])
    df = build_interaction_features(df, config)

    target = config["target"]["column"]

    feature_groups = {
        "clinical": (
            config["features"]["clinical"]
        ),

        "clinical_doppler": (
            config["features"]["clinical"]
            + config["features"]["doppler"]
        ),

        "clinical_doppler_biomarkers": (
            config["features"]["clinical"]
            + config["features"]["doppler"]
            + config["features"]["biomarkers"]
        ),
        
        "clinical_doppler_biomarkers_interactions": (
            config["features"]["clinical"]
            + config["features"]["doppler"]
            + config["features"]["biomarkers"]
            + config["features"]["interaction_features"]
        ),
    }

    models = get_models()

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    scoring = {
        'roc_auc': 'roc_auc',                    # Standard binary ROC-AUC
        'f1': make_scorer(f1_score, average='weighted'),
        'accuracy': 'accuracy',                  # Standard global accuracy
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted')
    }

    results = []

    for feature_set_name, feature_list in feature_groups.items():

        available = [f for f in feature_list if f in df.columns]

        X = df[available]
        y = df[target]
        
        y = y.isin(
            ["PE", "IUGR+PE"]
        ).astype(int)   
        

        print(f"\nTesting feature set: {feature_set_name}")

        for model_name, model in models.items():
            
            if (feature_set_name == "clinical_doppler_biomarkers_interactions"):
                pipeline = Pipeline([
                    ("plgf_doppler_discordance", PlGFDopplerDiscordance()),
                    ("placental_score", PlacentalDysfunctionScore()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", model),
                ])
            else:
                pipeline = Pipeline([
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

            results.append({
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Num_Features": len(available),

                "ROC_AUC_Mean": scores["test_roc_auc"].mean(),
                "ROC_AUC_STD": scores["test_roc_auc"].std(),

                "F1_Mean": scores["test_f1"].mean(),
                "F1_STD": scores["test_f1"].std(),

                "Accuracy_Mean": scores["test_accuracy"].mean(),
                "Precision_Mean": scores["test_precision"].mean(),
                "Recall_Mean": scores["test_recall"].mean(),
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["Feature_Set", "ROC_AUC_Mean"],
        ascending=[True, False]
    )

    Path("results").mkdir(exist_ok=True)

    results_df.to_csv(
        "results/feature_set_model_comparison.csv",
        index=False
    )

    print(results_df)


if __name__ == "__main__":
    main()