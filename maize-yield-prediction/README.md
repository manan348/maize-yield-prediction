# 🌽 Maize Yield Prediction System

> Predict maize grain yield for hybrid crosses across multiple locations using **Random Forest**, **genomic SNPs**, and **environmental + weather features**. Models **G×E (Genotype × Environment) interactions** to guide breeders in selecting optimal parent combinations and growing locations.

---

👤 **Author:** Abdul Manan — Plant Breeder | ML/DL Researcher  
📧 [abdulmanan2287@gmail.com](mailto:abdulmanan2287@gmail.com) &nbsp;|&nbsp; 🔗 [LinkedIn](https://www.linkedin.com/in/abdul-manan-0aa546332/) &nbsp;|&nbsp; 💻 [GitHub](https://github.com/manan348)  
🗓️ **Last Updated:** March 2026

---

## 📋 Table of Contents

- [🌽 Maize Yield Prediction System](#-maize-yield-prediction-system)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [📂 Dataset](#-dataset)
  - [📊 Model Performance](#-model-performance)
    - [✅ Final Model — v5 (Random Forest)](#-final-model--v5-random-forest)
  - [🔍 Feature Importance](#-feature-importance)
  - [🧠 Pipeline Overview](#-pipeline-overview)
    - [1️⃣ Data Preprocessing](#1️⃣-data-preprocessing)
    - [2️⃣ Feature Engineering](#2️⃣-feature-engineering)
    - [3️⃣ Model Training](#3️⃣-model-training)
    - [4️⃣ Prediction System](#4️⃣-prediction-system)
  - [🌽 Example Predictions](#-example-predictions)
    - [Single-Cross Predictions](#single-cross-predictions)
    - [Top Locations for B73 × Mo17](#top-locations-for-b73--mo17)
  - [📊 Visualizations](#-visualizations)
  - [⚙️ Requirements](#️-requirements)
  - [🚀 Installation](#-installation)
  - [🖥️ Running Modes](#️-running-modes)
    - [1) Train mode](#1-train-mode)
    - [2) Predict mode](#2-predict-mode)
    - [3) Streamlit app](#3-streamlit-app)
  - [📦 Artifacts Produced](#-artifacts-produced)
    - [`outputs/predictions/`](#outputspredictions)
    - [`outputs/plots/`](#outputsplots)
  - [📁 Project Structure](#-project-structure)
  - [📄 License](#-license)

---

## 🎯 Overview

This project builds a **genomic prediction system** for maize grain yield using SNP markers, plant traits, and weather features. The goal is to help breeders identify the **best parent crosses** and **optimal growing locations** before committing to field trials — reducing cost and time-to-insight.

The system supports three execution paths:

- **Full training pipeline** — runs when the raw H5 genotype file is available
- **Predict mode** — runs inference from saved model artifacts
- **Lightweight fallback** — serves precomputed predictions when heavy artifacts or H5 data are missing

Key capabilities:

- Predicts yield (bu/A) for any `(female × male, location)` combination
- Classifies predictions into **High / Medium / Low** categories
- Models **G×E interactions** across diverse environments
- Gracefully degrades to fallback mode without user intervention

---

## 📂 Dataset

| Field | Details |
|-------|---------|
| **Source** | [G2F 2017 — CyVerse Data Commons](https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_2014_2017_v1/G2F_Planting_Season_2017_v1) |
| **License** | Public research dataset |

**Local data files used by default:**

| File | Notes |
|------|-------|
| `data/g2f_2017_hybrid_data_clean.csv` | Phenotype and plant trait data |
| `data/g2f_2017_weather_data.csv` | Weather observations by location and date |
| `data/g2f_2017_ZeaGBSv27_Imputed_AGPv4.h5` | Genotype SNP matrix *(optional — often excluded from repo)* |

> If the genotype H5 file is missing, the pipeline automatically falls back to precomputed predictions.

---

## 📊 Model Performance

Each version builds incrementally on the previous, demonstrating the value of each added data source:

| Version | Change | CV R² | Test R² |
|---------|--------|:-----:|:-------:|
| v1 | Raw SNPs only | neg | 0.146 |
| v2 | + Environment | 0.111 | 0.487 |
| v3 | + PCA compression | 0.169 | 0.490 |
| v4 | + Weather data | 0.183 | 0.508 |
| **v5** | **No averaging (best)** | **0.572** | **0.635** |

### ✅ Final Model — v5 (Random Forest)

| Metric | Value |
|--------|-------|
| Test R² | **0.635** |
| CV Mean R² | **0.572** |
| Training Samples | 2,867 |
| Error Metric | MAE (bu/A) |

> **Reproducibility note:** Results are from experiment v5 on the G2F 2017 dataset split. Full model-evaluation metrics are written to `outputs/predictions/metrics.json` (train mode) and `outputs/predictions/inference_metrics.json` (predict mode).

> ⚠️ **`lightweight_metrics.json` is not a model-evaluation file.** It contains summary statistics derived from `all_predictions.csv`: row count, yield mean, std, min, and max only.

---

## 🔍 Feature Importance

| Category | Importance | Notes |
|----------|:----------:|-------|
| Plant Trait | **43.1%** | Reflects field-expressed genetic potential |
| Season Weather | **31.9%** | Growing-season climate aggregates |
| Genetics (PCA) | **18.4%** | 10 principal components from 10,000 SNPs |
| Critical Weather | **6.6%** | Flowering-period weather window |

> **Plant traits dominate** because they integrate both genetic and environmental influences as expressed in the field.

---

## 🧠 Pipeline Overview

### 1️⃣ Data Preprocessing
- Load phenotype CSV and match parent IDs
- Load genotype H5 and extract SNPs — 5,000 per female + 5,000 per male (10,000 total)
- Aggregate weather by growing season and critical flowering period
- Encode field locations

### 2️⃣ Feature Engineering
- PCA compresses 10,000 SNPs → **10 principal components**
- Combine genomic PCs with **17 environmental / plant trait features**
- Final feature vector: **27 features**

### 3️⃣ Model Training
- **Algorithm:** Random Forest Regressor
- **Hyperparameters:** `n_estimators=200` | `max_depth=10` | `min_samples_leaf=5`
- **Validation:** 5-fold cross-validation

### 4️⃣ Prediction System

```python
predict_yield(parent1, parent2, location)
# → Returns: Yield (bu/A) + Category (High / Medium / Low)
```

**Fallback outputs** (used when H5 data or trained model artifacts are unavailable):
- `outputs/predictions/all_predictions.csv`
- `outputs/predictions/lightweight_metrics.json`
- `outputs/predictions/top_crosses_by_location.csv`

---

## 🌽 Example Predictions

### Single-Cross Predictions

| Female | Male | Location | Yield (bu/A) | Category |
|--------|------|----------|:------------:|:--------:|
| B73 | Mo17 | ILH1 | 174.03 | 🟢 High |
| B73 | Mo17 | GAH1 | 108.83 | 🔴 Low |
| A632 | 3IIH6 | WIH1 | 178.64 | 🟢 High |
| Oh43 | Mo17 | IAH4 | 196.05 | 🟢 High |
| B97 | 3IIH6 | MNH1 | 151.04 | 🟡 Medium |

### Top Locations for B73 × Mo17

| Rank | Location | Yield (bu/A) |
|:----:|----------|:------------:|
| 1 | WIH2 | 218.88 |
| 2 | WIH1 | 211.77 |
| 3 | NYH3 | 203.59 |
| 4 | ONH1 | 201.79 |
| 5 | DEH1 | 201.62 |

---

## 📊 Visualizations

All plots are saved to `outputs/plots/`:

| File | Description |
|------|-------------|
| `actual_vs_predicted.png` | Predicted vs observed yield scatter |
| `feature_importance.png` | Importance grouped by category |
| `pca_variance.png` | Scree plot for genomic PCs |
| `residual_plot.png` | Residual distribution and patterns |
| `yield_distribution.png` | Yield histogram and by-location boxplots |
| `yield_vs_plant_height.png` | Trait correlation with yield |
| G×E Heatmap | Genotype × Environment yield interaction heatmap |
| Cross-Validation Scores | Fold-by-fold R² distribution |

---

## ⚙️ Requirements

| Library | Version |
|---------|---------|
| Python | 3.8+ |
| NumPy | ≥ 1.24.0 |
| Pandas | ≥ 2.0.0 |
| Scikit-learn | ≥ 1.3.0 |
| h5py | ≥ 3.9.0 |
| Matplotlib | latest |
| Plotly | latest |
| Streamlit | ≥ 1.28.0 |
| Joblib | latest |

> **Note:** Verify that `requirements.txt` is in standard pip format (one package per line) before running `pip install -r requirements.txt`. If it contains markdown-style content, install packages manually from the table above or convert the file first.

---

## 🚀 Installation

> Designed for **local Python** execution.

```bash
# Clone the repository
git clone https://github.com/manan348/maize-yield-prediction.git
cd maize-yield-prediction

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Running Modes

### 1) Train mode

```bash
python main.py --mode train
```

What it does:
- Loads `data/g2f_2017_hybrid_data_clean.csv` and `data/g2f_2017_weather_data.csv`
- Loads genotype H5 file if present at `data/g2f_2017_ZeaGBSv27_Imputed_AGPv4.h5`
- Builds 27-feature matrix (10 SNP PCA components + 17 environmental/trait features)
- Trains Random Forest with 5-fold cross-validation
- Saves model artifacts and evaluation metrics to `outputs/predictions/`

### 2) Predict mode

```bash
python main.py --mode predict
```

Behavior:
- If `model.joblib`, `X_final.npy`, and `y.npy` exist in `outputs/predictions/` → runs inference and writes:
  - `outputs/predictions/inference_predictions.csv`
  - `outputs/predictions/inference_metrics.json`
- Otherwise → switches to **lightweight fallback** and writes summary artifacts from `all_predictions.csv`

### 3) Streamlit app

```bash
streamlit run app/app.py
```

> The app is a **lookup and inference UI** over precomputed predictions. It does **not** run model training. It expects `outputs/predictions/all_predictions.csv` to be present.

---

## 📦 Artifacts Produced

### `outputs/predictions/`

| File | Generated By | Description |
|------|:------------:|-------------|
| `all_predictions.csv` | Precomputed / fallback | All cross × location yield predictions |
| `lightweight_metrics.json` | Fallback mode | Summary stats only: rows, yield mean/std/min/max |
| `top_crosses_by_location.csv` | Fallback mode | Ranked top crosses per location |
| `metrics.json` | Train mode | CV R², Test R², MAE ← model evaluation |
| `test_predictions.csv` | Train mode | Held-out test set predictions |
| `model.joblib` | Train mode | Serialized Random Forest model |
| `best_model.pkl` | Train mode | Best model checkpoint |
| `pca.pkl` | Train mode | Fitted PCA transformer |
| `scaler_snp.pkl` | Train mode | SNP feature scaler |
| `scaler_final.pkl` | Train mode | Final feature scaler |
| `X_snp.npy` | Train mode | SNP feature array |
| `X_env.npy` | Train mode | Environmental feature array |
| `X_final.npy` | Train mode | Combined feature matrix |
| `y.npy` | Train mode | Target yield array |
| `df_raw.csv` | Train mode | Processed raw dataframe |
| `inference_predictions.csv` | Predict mode | Inference output on full dataset |
| `inference_metrics.json` | Predict mode | Evaluation metrics from inference run ← model evaluation |

### `outputs/plots/`

Generated during training: `actual_vs_predicted.png`, `feature_importance.png`, `pca_variance.png`, `residual_plot.png`, `yield_distribution.png`, `yield_vs_plant_height.png`, and related outputs.

---

## 📁 Project Structure

```
maize-yield-prediction/
├── main.py                          # Primary CLI entrypoint (--mode train / predict)
├── app/
│   └── app.py                       # Streamlit lookup & inference UI
├── src/
│   ├── data/
│   │   ├── load_data.py             # H5 and CSV data loading
│   │   └── preprocess.py            # Phenotype cleaning, SNP extraction, weather aggregation
│   ├── features/
│   │   └── build_features.py        # PCA compression, feature assembly (27 features)
│   ├── models/
│   │   ├── train.py                 # Random Forest training + cross-validation
│   │   └── predict.py               # Inference logic + fallback handler
│   └── visualization/
│       └── plots.py                 # All plot generation
├── data/
│   ├── g2f_2017_hybrid_data_clean.csv
│   ├── g2f_2017_weather_data.csv
│   └── g2f_2017_ZeaGBSv27_Imputed_AGPv4.h5  # Optional — often excluded from repo
├── outputs/
│   ├── predictions/                 # All model outputs, metrics, and fallback CSVs
│   └── plots/                       # Generated visualizations
├── notebook/                        # Exploratory notebooks
├── requirements.txt
├── README.md
```

---

## 📄 License

**Dataset:** The G2F 2017 dataset is a public research dataset from CyVerse Data Commons. Please cite the original data source in any publications using this work.

**Code:** Released under the **MIT License** — applies if a `LICENSE` file is included and set accordingly. See `LICENSE` for details.

---

*Built  for plant breeders navigating the complexity of G×E interactions.*
