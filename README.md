# Preeclampsia AI Prediction Project

A modular Python project for preeclampsia prediction and Gestational Age at Delivery Prediction using clinical, uterine artery Doppler, and angiogenic biomarkers.

## Main research idea

**Doppler–Angiogenic Interaction Features for Explainable Preeclampsia Prediction and Gestational Age at Delivery Prediction**



The project supports:

* Data validation and preprocessing
* Biomarker interaction feature engineering
* Clinical, Doppler, and biomarker feature groups
* Feature ablation experiments
* Machine learning benchmarking
* SHAP-based explainability and its stability through cross-validation
* Gestational Age at Delivery Prediction
* Reproducible experiments

## Dataset

Place the original dataset in:

`data/raw/ljubljana.xlsx`

The current project is designed around variables such as:

* PlGF
* sFlt-1
* Uterine artery RI
* Uterine artery PI
* Mean RI
* Mean PI
* Clinical variables
* PE / IUGR / Control labels

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r packages.txt
```

## Run (Research experiments)



\# Models Comparison



```bash

python -m src.run_approach

```



This will produce:



\- `results/model_benchmark.csv`



\---



\# Ablation Study



```bash

python -m src.experiments.feature_interaction_study

```



This will produce:



\- `results/interaction_feature_importance.csv`



```bash

python -m src.experiments.ablation_study

```



This will produce:



\- `results/ablation_feature_sets.csv`



```bash

python -m src.models.evaluateFeatureSets

```



This will produce:



\- `results/feature_set_model_comparison.csv`



\---



\# Selective Prediction Performance



```bash

python -m src.models.selectivePrediction

```



This will produce:



\- `results/selective_prediction.csv`



\---



\# SHAP Analysis





```bash

python -m src.explainability.preeclampsia-shap-random-forest-cv

```
This will produce the following documents in `results/shap-random-forest-cv/`:



\- `feature-importance.csv`

\- `shap-contributions.csv`

\- `shap-global-bar.png`

\- `shap-global-beeswarm.png`

\- `shap-global-explanation.pkl`


```bash

python -m src.explainability.explainBestModel

```

This will produce the following documents in `results/shap/`:



\- `feature_importance.csv`

\- `shap_summary_bar.png`

\- `shap_summary_beeswarm.png`

\- `dependence_<feature>.png`

\- `waterfall_<patient>.png`





```bash

python -m src.explainability.shap-stability-analysis

```



This will produce the following documents in `results/shap-cross-val/`:



\- `shap_fold_importance.csv`

\- `shap_stability_summary.csv`

\- `shap_fold_ranks.csv`

\- `shap_rank_correlation.csv`

\- `shap_importance_mean.png`

\- `shap_importance_stability.png`





```bash

python -m src.explainability.explainFeatureInteractions

```



This will produce the results in:



\- `results/feature_interactions/`





```bash

python -m src.explainability.shap-interaction-stability

```



This will produce the following documents in `results/shap-interaction-stability/`:



\- `shap_interaction_split_values.csv`

\- `shap_interaction_stability_summary.csv`

\- `shap_interaction_split_ranks.csv`

\- `shap_interaction_rank_correlation.csv`

\- `shap_interaction_jaccard.csv`

\- `shap_interaction_heatmap.csv`

\- `shap_interaction_report.txt`

\- `shap_interaction_stability.png`

\- `shap_interaction_heatmap.png`





```bash

python -m src.explainability.local-shap-stability

```



This will produce the following documents in `results/local_shap_stability/`:



\- `local_shap_values.csv`

\- `local_shap_summary.csv`

\- `local_top_features.csv`

\- `local_prediction_stability.csv`

\- `local_patient_stability.csv`

\- `local_jaccard.csv`

\- `local_rank_correlation.csv`

\- `local_shap_stability_report.txt`

\- `local_explanation_stability.png`

\- `local_probability_stability.png`



\---



\# Gestational Age at Delivery Prediction



```bash

python -m src.experiments.gestationalAgeRegression

```



This will produce:



\- `gestational_age_regression.csv`

\- `prediction_vs_actual.png`

\- `residual_plot.png`

\- `residual_histogram.png`

