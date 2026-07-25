from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config
from src.models.evaluateFeatureSets import get_models

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

    features = [f for f in features if f in df.columns]

    X = df[features]

    y = df[target]
    
    y = y.isin(
            ["PE", "IUGR+PE"]
        ).astype(int)

    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1"
    }

    results = []

    best_model = None
    best_name = None
    best_auc = -1
    best_f1 = -1

    print("\nEvaluating models...\n")

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
            scoring=scoring,
            n_jobs=-1
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