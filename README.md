# Preeclampsia AI Prediction Project

A modular Python project for early preeclampsia prediction and Gestational Age at Delivery Prediction using clinical, uterine artery Doppler, and angiogenic biomarkers.

## Main research idea

**Doppler–Angiogenic Interaction Features for Explainable Early Preeclampsia Prediction and Gestational Age at Delivery Prediction**



The project supports:

* Data validation and preprocessing
* Biomarker interaction feature engineering
* Clinical, Doppler, and biomarker feature groups
* Feature ablation experiments
* Machine learning benchmarking
* SHAP-based explainability
* Gestational Age at Delivery Prediction
* Reproducible experiments

## Dataset

Place the original dataset in:

`data/raw/ljubljana.csv`

The current project is designed around variables such as:

* PlGF
* sFlt-1
* Uterine artery RI
* Uterine artery PI
* Mean RI
* Mean PI
* Clinical variables
* PE / IUGR / Control labels

> Check the exact column names in your dataset before running the pipeline.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\Scripts\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (Research experiments)



\# Models Comparison



```bash

python -m src.run\_approach

```



This will produce:



\- `results/model\_benchmark.csv`



\---



\# Ablation Study



```bash

python -m src.experiments.feature\_interaction\_study

```



This will produce:



\- `results/interaction\_feature\_importance.csv`



```bash

python -m src.experiments.ablation\_study

```



This will produce:



\- `results/ablation\_feature\_sets.csv`



```bash

python -m src.models.evaluateFeatureSets

```



This will produce:



\- `results/feature\_set\_model\_comparison.csv`



\---



\# Selective Prediction Performance



```bash

python -m src.models.slectivePrediction

```



This will produce:



\- `results/selective\_prediction.csv`



\---



\# SHAP Analysis



```bash

python -m src.explainability.explainBestModel

```



This will produce the following files in `results/shap/`:



\- `feature\_importance.csv`

\- `shap\_summary\_bar.png`

\- `shap\_summary\_beeswarm.png`

\- `dependence\_<feature>.png`

\- `waterfall\_<patient>.png`



```bash

python -m src.explainability.explainFeatureInteractions

```



This will produce the results in:



\- `results/feature\_interactions/`



\---



\# Gestational Age at Delivery Prediction



```bash

python -m src.experiments.gestationalAgeRegression

```



This will produce:



\- `gestational\_age\_regression.csv`

\- `prediction\_vs\_actual.png`

\- `residual\_plot.png`

\- `residual\_histogram.png`





