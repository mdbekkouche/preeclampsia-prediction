from pathlib import Path
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Please place your CSV file in data/raw/."
        )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Supported formats: CSV, XLSX, XLS")

     # Row 1 contains the real variable names
    df.columns = df.iloc[1]

    # Rename the first column because it is the label
    df = df.rename(
        columns={
            df.columns[0]: "PE_Label"
        }
    )

    # Remove the two header rows
    df = df.iloc[2:].reset_index(drop=True)
    
    # Remove line 29, 61, 73, 95
    df = df.drop(index=29).reset_index(drop=True)
    df = df.drop(index=61).reset_index(drop=True)
    df = df.drop(index=73).reset_index(drop=True)
    df = df.drop(index=95).reset_index(drop=True)
    
    df = df.drop(columns=["Patient nuber"])
    #df = df.drop(columns=["Gestational age at delivery [weeks]"])
    #df = df.drop(columns=["Birth weight [g]"])
    #df = df.drop(columns=["PE_Label"])
    
    print(df["S-PLGF [µg/L]"])
    
    # Remove unnamed or empty columns
    df = df.loc[
        :,
        ~df.columns.isna()
    ]

    df = df.loc[
        :,
        ~df.columns.astype(str).str.startswith("Unnamed")
    ]

    print(f"Dataset shape: {df.shape}")

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset columns:")
    print(df.columns.tolist())
    
    nan_rows = df[df.isna().any(axis=1)]

    print(nan_rows)
    print("Indexes of rows with NaN values:")
    print(nan_rows.index)
    
    return df
