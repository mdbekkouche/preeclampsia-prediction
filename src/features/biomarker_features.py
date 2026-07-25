import numpy as np


def create_angiogenic_imbalance_index(df):
    """log(sFlt-1 / PlGF)"""
    df["log_sFlt1_PlGF"] = np.log(
        (df["S-Flt1 [µg/L]"] + 1e-8) / (df["S-PLGF [µg/L]"] + 1e-8)
    )
    return df


def create_doppler_angiogenic_index(df):
    """Mean PI * log(sFlt-1 / PlGF)"""
    df["Doppler_Angiogenic_Index"] = (
        df["Mean PI"] * df["log_sFlt1_PlGF"]
    )
    return df


def create_placental_dysfunction_score(df):
    """z(Mean PI) + z(sFlt-1) - z(PlGF)"""
    for col in ["Mean PI", "S-Flt1 [µg/L]", "S-PLGF [µg/L]"]:
        z_col = f"z_{col.replace('-', '_').replace(' ', '_')}"
        df[z_col] = (df[col] - df[col].mean()) / df[col].std()

    df["Placental_Dysfunction_Score"] = (
        df["z_Mean_PI"] 
        + df["z_S_Flt1_[µg/L]"]
        - df["z_S_PLGF_[µg/L]"] 
    )

    return df


def create_vascular_placental_index(df):
    """MAP * log(sFlt-1 / PlGF)"""
    df["Vascular_Placental_Index"] = (
        df["MAP"] * df["log_sFlt1_PlGF"]
    )
    return df


def create_map_doppler_index(df):
    """MAP * Mean PI"""
    df["MAP_Doppler_Index"] = df["MAP"] * df["Mean PI"]
    return df


def create_plgf_doppler_discordance(df):
    """z(Mean PI) - z(PlGF)"""
    z_pi = (df["Mean PI"] - df["Mean PI"].mean()) / df["Mean PI"].std()
    z_plgf = (df["S-PLGF [µg/L]"] - df["S-PLGF [µg/L]"].mean()) / df["S-PLGF [µg/L]"].std()

    df["PlGF_Doppler_Discordance"] = z_pi - z_plgf
    return df
