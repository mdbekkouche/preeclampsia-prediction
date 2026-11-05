from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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

def run_model_benchmark(df: pd.DataFrame, config: dict):
    target_column = config["target"]["column"]

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    X = df.drop(columns=[target_column])
    
    df[target_column] = df[target_column].isin(
        ["PE", "IUGR+PE"]
    ).astype(int)
    
    y = df[target_column]
    
    print(y.value_counts())
    
    #X = X.select_dtypes(include=["number"])
    
    X = X.apply(pd.to_numeric, errors='coerce')
    
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
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
        
        results.append({
            "Model": name,
            "ROC_AUC_Mean": scores["test_roc_auc"].mean(),
            "ROC_AUC_STD": scores["test_roc_auc"].std(),
            "F1_Mean": scores["test_f1"].mean(),
            "F1_STD": scores["test_f1"].std(),
        })

    results_df = pd.DataFrame(results)

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(
        "results/model_benchmark.csv",
        index=False
    )

    print(results_df)
