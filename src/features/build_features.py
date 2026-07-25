import pandas as pd

from src.features.biomarker_features import (
    create_angiogenic_imbalance_index,
    create_doppler_angiogenic_index,
    create_placental_dysfunction_score,
    create_vascular_placental_index,
    create_map_doppler_index,
    create_plgf_doppler_discordance,
)

from src.features.doppler_features import (
    create_pi_difference,
    create_ri_difference,
)


def build_interaction_features(df, config):
    target_column = config["target"]["column"]
    
    cols = df.columns.difference([target_column])

    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    df = df.copy()

    df = create_angiogenic_imbalance_index(df)
    df = create_doppler_angiogenic_index(df)
    df = create_placental_dysfunction_score(df)
    #df = create_vascular_placental_index(df)
    #df = create_map_doppler_index(df)
    df = create_plgf_doppler_discordance(df)
    #df = create_pi_difference(df)
    #df = create_ri_difference(df)

    return df
