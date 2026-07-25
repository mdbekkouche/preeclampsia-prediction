from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, roc_auc_score, f1_score


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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, average="weighted")
    }

    results = []

    for name, model in models.items():
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
            scoring=scoring
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
