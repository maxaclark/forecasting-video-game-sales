# Forecasting Video Game Sales 

A lightweight machine‑learning pipeline and Streamlit app to **forecast global video‑game sales** from game metadata (title, genre, platform, publisher, release year, franchise tags, etc.).
The work is split into: 
- **Data cleaning & feature engineering** (notebooks) 
-  **Exploratory Data Analysis (EDA)** 
-   **Modeling pipeline** (NLP + tabular features) 
-    **Prototype Streamlit dashboard** for visualization and “what-if” exploration

> **Project status:** *Work in progress*. The Streamlit dashboard is partially implemented; model integration and advanced feature insights will land in the next milestone.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
  - [1) Create a virtual environment (optional)](#1-create-a-virtual-environment-optional)
  - [2) Install dependencies](#2-install-dependencies)
  - [3) Launch the app](#3-launch-the-app)
  - [4) Stop the app](#4-stop-the-app)
- [Using the App](#using-the-app)
  - [Sales Trends (Working)](#-sales-trends-working)
  - [Prediction Interface (UI Only)](#-prediction-interface-ui-only)
  - [Feature Insights (Planned)](#-feature-insights-planned)


---

## Repository Structure
```text
forecasting-video-game-sales/
├── data/
│   ├── X_test_name.npy
│   ├── X_train_name.npy
│   ├── test_results.csv
│   ├── title_embeddings.parquet         # Text embeddings used during training
│   ├── train_test_split.npz             # Train/test arrays for modeling
│   ├── vgsales.csv                      # Original Kaggle dataset
│   ├── vgsales_clean.csv                # Cleaned dataset for dashboard
│   └── vgsales_consolidated.csv         # Consolidated (franchise/title-level) dataset
│
├── models/
│   ├── knn_model.joblib
│   ├── lightgbm_model.joblib
│   ├── nn_model.keras
│   ├── svm_model.joblib
│   └── xgb_model.joblib
│
├── notebooks/
│   ├── ML_pipeline.ipynb                # Model training workflow
│   ├── Vgsales_EDA.ipynb                # Exploratory Data Analysis
│   └── platform_consolidation.ipynb     # Feature engineering
│
├── src/
│   └── data_cleaning.py                 # Cleaning & preprocessing scripts
│
├── visualizations/
│   ├── lgbm_feature_importance.png
│   ├── nn_feature_importance.png
│   ├── nn_loss_mae.png
│   └── xgb_feature_importance.png
│
├── app.py                               # Streamlit Dashboard (prototype)
└── README.md
```

---

## Requirements
- **Python** 3.8+
- **pip**
- Local clone of this repository
- `data/vgsales_clean.csv` available locally (see structure above)

---

## Quickstart

### 1) Create a virtual environment (optional)
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install streamlit pandas plotly
```

### 3) Launch the app
```bash
streamlit run app.py
```
Then open your browser at:
```
http://localhost:8501
```

### 4) Stop the app
Press **Ctrl + C** in the terminal running Streamlit.

---

## Using the App

###  Sales Trends (Working)
- Filter by **Year**, **Genre**, **Platform**
- Aggregate by **Year / Genre / Platform**
- Interactive charts + **Top games** table

###  Prediction Interface (UI Only)
- Fill in metadata: **Title, Platform, Genre, Year, Publisher**
- Form submission is wired; **model predictions pending**
- Regression model integration will arrive in the next milestone

###  Feature Insights (Planned)
Intended to include:
- Feature importance visualizations
- Correlation heatmaps

---
