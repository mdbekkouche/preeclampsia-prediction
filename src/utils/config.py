import yaml


def load_config(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
