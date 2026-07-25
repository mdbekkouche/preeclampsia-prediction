from pathlib import Path
import pandas as pd

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config


def main():
    config = load_config("config/config.yaml")
    df = load_dataset(config["data"]["raw_file"])
    df = build_interaction_features(df, config)

    target = config["target"]["column"]

    feature_groups = {
        "clinical": config["features"]["clinical"],
        "clinical-doppler": (
            config["features"]["clinical"]
            + config["features"]["doppler"]
        ),
        "clinical-doppler-biomarkers": (
            config["features"]["clinical"]
            + config["features"]["doppler"]
            + config["features"]["biomarkers"]
        ),
        "clinical-doppler-biomarkers-interaction-features": (
            config["features"]["clinical"]
            + config["features"]["doppler"]
            + config["features"]["biomarkers"]
            + config["features"]["interaction_features"]
        )
    }

    rows = []

    for name, features in feature_groups.items():
        available = [f for f in features if f in df.columns]

        rows.append({
            "Feature_Set": name,
            "Number_of_Features": len(available),
            "Features": ", ".join(available)
        })

    result = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)
    result.to_csv(
        "results/ablation_feature_sets.csv",
        index=False
    )

    print(result)


if __name__ == "__main__":
    main()
