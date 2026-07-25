import pandas as pd

from src.features.biomarker_features import (
    create_angiogenic_imbalance_index,
)


def test_angiogenic_index():
    df = pd.DataFrame({
        "sFlt-1": [100.0],
        "PlGF": [10.0],
    })

    result = create_angiogenic_imbalance_index(df)

    assert "log_sFlt1_PlGF" in result.columns
    assert result["log_sFlt1_PlGF"].notna().all()
