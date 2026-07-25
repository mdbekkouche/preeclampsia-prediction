import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: list):
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"- {c}" for c in missing)
        )


def report_missing_values(df: pd.DataFrame):
    report = df.isna().sum()
    return report[report > 0].sort_values(ascending=False)
