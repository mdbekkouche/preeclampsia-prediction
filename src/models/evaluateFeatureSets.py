from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config


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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall"
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
                n_jobs=-1
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