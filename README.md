# creditcard_psp

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
# CreditCard PSP Routing – Machine Learning Pipeline

This repository implements a **Payment Service Provider (PSP) routing model** to optimize transaction success rates and processing costs.  
It follows the **CRISP-DM** framework and uses a modular Cookiecutter structure for reproducibility and deployment.

---

## 🧠 Project Overview

The model predicts PSP success probabilities and expected fees per transaction, then computes a **business score** balancing success vs. cost using a configurable `alpha` parameter.

**Core features:**
- Transaction preprocessing & feature engineering (`creditcard_psp/features.py`)
- Model training & calibration (LogReg, RF, XGBoost)
- Evaluation (ROC, PR-Curves, Gain Chart, SHAP)
- Dashboard bundle for real-time routing
- Makefile-based automation

---

## ⚙️ Setup

### Requirements
- **Python 3.11** (recommended)
- pip / pipx
- optional: `make`

### Installation
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### 1️⃣ Train Models
```bash
make train
```
Trains and calibrates all models, storing them in `models/`.

### 2️⃣ Evaluate
```bash
make evaluate
```
Generates reports, calibration plots, and SHAP visualizations in `reports/`.

### 3️⃣ Bundle for Dashboard
```bash
make bundle
```
Exports all required artifacts into `dashboard_data/`:
- `psp_router_model.joblib`
- `model_raw.joblib`
- `df.parquet`, `X_te.parquet`
- `psps.json`, `P_matrix.npy`

### 4️⃣ Predict (inference)
```bash
make predict
```
Performs scoring using the saved model bundle.

Without `make`:
```bash
python -m creditcard_psp.modeling.train
python -m creditcard_psp.modeling.predict --input data/sample_input.csv --out predictions.csv
```

---

## 📊 Business Score Formula

The model computes a cost-adjusted score per PSP:

```python
score = P - alpha * expected_cost 
```

- `alpha ∈ [0,1]` controls the weight between cost vs. success.
- Score is calculated in the Dashboard in app.py
- PSP metadata & fees are loaded from `dashboard_data/psps.json` and e.g. `data/raw/PSP_Servicegebuehren.xlsx`.

---

## 🧪 Testing

Install dev dependencies and run smoke tests:
```bash
pytest -q
```

---

## 📁 Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         creditcard_psp and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── creditcard_psp   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes creditcard_psp a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

## 🔒 Reproducibility
- Python 3.11
- Pinned dependencies in `requirements.txt`
- Metrics saved in `reports/summary_metrics.csv`
- Artifacts bundled in `dashboard_data/`

---

## 🧾 License
MIT License © 2025 Miriam Mantiuk
