# 🩸 Blood Cell Anomaly Detection

**Machine Learning Classification Project — Streamlit App + FastAPI Service**

An end-to-end ML project that predicts whether a blood cell is **Normal** or **Anomalous** from 26 biological, imaging, and lab-test features. Includes a full Jupyter notebook (EDA → training → model comparison), an interactive **Streamlit** app for live predictions, and a **FastAPI** REST service for programmatic access.

🔗 **Live Demo:** [mycleanapp.streamlit.app](https://mycleanapp-cwfiw7gku84py84fhjsappx.streamlit.app/)

---

## 📋 Project Overview

The goal is to classify blood cells into:

| Value | Class |
|---|---|
| `0` | Normal |
| `1` | Anomaly |

The dataset contains **5,880 samples** (4,000 Normal / 1,880 Anomaly).

The notebook (`Blood_Cell_Anomaly_Detection.ipynb`) covers the full workflow:

1. Import libraries and load the dataset
2. Data quality checks (missing values / duplicates)
3. Target distribution analysis (`anomaly_label`)
4. Feature selection and cleaning (dropping columns that would cause data leakage)
5. Encoding categorical features
6. Outlier inspection (without removal)
7. Correlation analysis
8. Stratified train/test split to preserve class balance
9. Feature scaling (`StandardScaler`, fit only on training data)
10. Training and evaluating 9 classification models
11. Model comparison and selection of the best performer

---

## 📁 Project Structure

```
.
├── app.py                                # Interactive Streamlit app
├── api.py                                # FastAPI REST service
├── Blood_Cell_Anomaly_Detection.ipynb    # Notebook: EDA, training, evaluation
├── blood_cell_anomaly_detection.csv      # Raw dataset (5,880 rows)
├── best_model.pkl                        # Final trained model (XGBoost, pickle format)
├── best_model.json                       # Same model in XGBoost's native JSON format
├── scaler.pkl                            # StandardScaler fitted on training data
├── feature_order.pkl                     # Exact list/order of the 26 input features
├── requirements.txt                      # Project dependencies
└── README.md                             # This file
```

> `best_model.json` is loaded by `api.py` and `app.py` instead of the pickle file, since loading an XGBoost model from its native JSON format is more robust across library/version changes than unpickling.

---

## 🧬 Features Used for Prediction

The model relies on **26 features** (see `feature_order.pkl` for the exact order expected by the model), grouped as follows:

**Cell morphology (11):**
`cell_diameter_um`, `nucleus_area_pct`, `chromatin_density`, `cytoplasm_ratio`, `circularity`, `eccentricity`, `granularity_score`, `lobularity_score`, `membrane_smoothness`, `cell_area_px`, `perimeter_px`

**Color / staining (4):**
`mean_r`, `mean_g`, `mean_b`, `stain_intensity`

**Lab blood test values (7):**
`wbc_count_per_ul`, `rbc_count_millions_per_ul`, `hemoglobin_g_dl`, `hematocrit_pct`, `platelet_count_per_ul`, `mcv_fl`, `mchc_g_dl`

**Imaging settings (2):**
`magnification_x`, `image_resolution_px`

**Patient data (2):**
`patient_age_group` (0 = Pediatric, 1 = Adult, 2 = Elderly), `patient_sex` (0 = Female, 1 = Male)

> ⚠️ The following raw columns were **excluded** from training to avoid target leakage or because they are metadata only: `cell_id`, `disease_category`, `cell_type`, `dataset_source`, `staining_protocol`, `microscope_model`, `cytodiffusion_anomaly_score`, `cytodiffusion_classification_confidence`, `labeller_confidence_score`.

---

## 🤖 Model Comparison

Nine classification algorithms were trained and evaluated on the same train/test split:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **XGBoost** ⭐ | **0.9787** | 0.9861 | 0.9468 | 0.9661 |
| Random Forest | 0.9694 | 0.9885 | 0.9149 | 0.9503 |
| Gradient Boosting | 0.9677 | 0.9913 | 0.9069 | 0.9472 |
| Decision Tree | 0.9260 | 0.8753 | 0.8963 | 0.8857 |
| SVM | 0.9209 | 0.9965 | 0.7553 | 0.8593 |
| KNN | 0.9090 | 0.9590 | 0.7473 | 0.8401 |
| AdaBoost | 0.9005 | 0.9331 | 0.7420 | 0.8267 |
| Logistic Regression | 0.8444 | 0.8386 | 0.6356 | 0.7231 |
| Naive Bayes | 0.7789 | 0.7117 | 0.5186 | 0.6000 |

**Selected model for deployment:** `XGBoost`, with **97.87%** accuracy — saved as `best_model.pkl` / `best_model.json`, together with `scaler.pkl` for feature scaling.

> Note: model performance is specific to this dataset and test split, and should not be generalized as an absolute benchmark without further external validation.

---

## 💻 Streamlit App (`app.py`)

Three sections navigable from the sidebar:

### 1. Live Prediction
- Manually input all 26 features from the sidebar
- Click **Predict Anomaly Status**
- View the result (✅ Normal / ⚠️ Anomaly) with the model's confidence score

### 2. Model Comparison
- Interactive table comparing all 9 models (Accuracy, Precision, Recall, F1)

### 3. Visualizations
- Bar chart comparing model accuracies
- Distribution plot for a sample of the `cell_diameter_um` feature

Run it with:
```bash
streamlit run app.py
```

---

## 🌐 REST API (`api.py`)

A FastAPI service exposing the same model for programmatic use.

Run it with:
```bash
uvicorn api:app --reload
```

Interactive docs available at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info, feature order, and encoding reference |
| `GET` | `/features` | Returns the 26 required features and their order |
| `POST` | `/predict` | Runs a prediction on a single sample |

### Example request

You can send either a raw ordered `features` list or the named fields:

```json
{
  "features": [10.18, 43.54, 0.39, 0.56, 0.77, 0.37, 1.88, 1.77, 0.84,
               336.57, 64.07, 212.12, 146.39, 168.66, 0.62, 7043.27,
               4.79, 13.55, 41.02, 249792.62, 88.94, 33.50, 76.01,
               336.24, 1, 0]
}
```

### Example response

```json
{
  "status": "success",
  "prediction_code": 0,
  "result": "Normal Cell",
  "confidence_pct": 98.42,
  "probabilities": { "normal": 0.9842, "anomaly": 0.0158 }
}
```

---

## ⚙️ Installation & Local Setup

### Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=1.7.0
```

> If you plan to run `api.py`, also install `fastapi` and `uvicorn` (not currently pinned in `requirements.txt`).

### Steps to Run

```bash
# 1) Clone the repo or download all files into one folder
git clone <repo-url>
cd <repo-folder>

# 2) Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the Streamlit app
streamlit run app.py

# and/or run the API
uvicorn api:app --reload
```

The Streamlit app opens at `http://localhost:8501`; the API docs are at `http://localhost:8000/docs`.

> Make sure `best_model.json`, `scaler.pkl`, and `feature_order.pkl` are in the same folder as `app.py` / `api.py`.

---

## 🚀 Deployment

The Streamlit app is deployed via **Streamlit Community Cloud**:

👉 https://mycleanapp-cwfiw7gku84py84fhjsappx.streamlit.app/

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas / NumPy** — data processing
- **Scikit-learn** — modeling and evaluation (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, SVM, KNN, Naive Bayes)
- **XGBoost** — final deployed model
- **Matplotlib / Seaborn** — exploratory data analysis
- **Streamlit** — interactive web app
- **FastAPI / Uvicorn** — REST API service
- **Joblib** — saving/loading the scaler and feature order

---

## 📌 Notes

- Outliers in the blood-test variables were inspected but not removed, since they may represent real observations.
- `StandardScaler` was fit only on training data to avoid leaking information from the test set.
- The train/test split was stratified to preserve class balance.

---

## 📄 License

This project is for educational and demonstration purposes only. Model outputs should not be relied upon as an actual medical diagnostic tool.
