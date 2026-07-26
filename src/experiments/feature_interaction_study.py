import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.utils.config import load_config


def main():
    config = load_config("config/config.yaml")

    df = load_dataset(config["data"]["raw_file"])
    df = df.drop(columns=["Gestational age at delivery [weeks]"])
    df = df.drop(columns=["Birth weight [g]"])
    df = build_interaction_features(df, config)

    target = config["target"]["column"]

    interaction_features = [
        "log_sFlt1_PlGF",
        "Doppler_Angiogenic_Index",
        "Placental_Dysfunction_Score",
        #"Vascular_Placental_Index",
        #"MAP_Doppler_Index",
        "PlGF_Doppler_Discordance",
        "PI_Difference",
        "RI_Difference",
    ]

    available = [
        col for col in interaction_features
        if col in df.columns
    ]

    X = df[available].fillna(df[available].median())
    y = df[target]

    scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    result = pd.DataFrame({
        "Feature": available,
        "Mutual_Information": scores
    }).sort_values(
        "Mutual_Information",
        ascending=False
    )

    result.to_csv(
        "results/interaction_feature_importance.csv",
        index=False
    )

    print(result)


if __name__ == "__main__":
    main()
