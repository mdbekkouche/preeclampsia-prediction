from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config
from src.models.evaluateFeatureSets import get_models   # reuse your models

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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
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