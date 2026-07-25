def create_pi_difference(df):
    """Absolute difference between right and left uterine artery PI."""
    df["PI_Difference"] = (
        df["Art ut. D-pulsatility index [PI]"]
        - df["Art ut. L-pulsatility index [PI]"]
    ).abs()

    return df


def create_ri_difference(df):
    """Absolute difference between right and left uterine artery RI."""
    df["RI_Difference"] = (
        df["Art ut. D-resistance index [RI]"]
        - df["Art ut. L-resistance index [RI]"]
    ).abs()

    return df
