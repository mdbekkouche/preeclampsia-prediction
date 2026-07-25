from pathlib import Path

from src.data.load_data import load_dataset
from src.features.build_features import build_interaction_features
from src.models.train_models import run_model_benchmark
from src.utils.config import load_config


def main():
    config = load_config("config/config.yaml")

    print("Loading dataset...")
    df = load_dataset(config["data"]["raw_file"])

    #print("Building interaction features...")
    df = build_interaction_features(df, config)

    print("Running model benchmark...")
    run_model_benchmark(df, config)

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
