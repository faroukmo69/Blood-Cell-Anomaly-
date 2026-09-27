from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

app = FastAPI(
    title="Blood Cell Anomaly Detection API",
    version="1.0",
    description="Predict whether a blood cell is Normal (0) or Anomalous (1) from 26 features.",
)

BASE_DIR = Path(__file__).resolve().parent

# تحميل من JSON (أضمن من pkl)
model = XGBClassifier()
model.load_model(str(BASE_DIR / "best_model.json"))

scaler = joblib.load(BASE_DIR / "scaler.pkl")
FEATURE_ORDER = joblib.load(BASE_DIR / "feature_order.pkl")
N_FEATURES = len(FEATURE_ORDER)  # 26


class BloodSample(BaseModel):
    features: list[float] | None = Field(
        default=None,
        description=f"List of exactly {N_FEATURES} floats in FEATURE_ORDER",
    )
    cell_diameter_um: float | None = None
    nucleus_area_pct: float | None = None
    chromatin_density: float | None = None
    cytoplasm_ratio: float | None = None
    circularity: float | None = None
    eccentricity: float | None = None
    granularity_score: float | None = None
    lobularity_score: float | None = None
    membrane_smoothness: float | None = None
    cell_area_px: float | None = None
    perimeter_px: float | None = None
    mean_r: float | None = None
    mean_g: float | None = None
    mean_b: float | None = None
    stain_intensity: float | None = None
    wbc_count_per_ul: float | None = None
    rbc_count_millions_per_ul: float | None = None
    hemoglobin_g_dl: float | None = None
    hematocrit_pct: float | None = None
    platelet_count_per_ul: float | None = None
    mcv_fl: float | None = None
    mchc_g_dl: float | None = None
    magnification_x: float | None = None
    image_resolution_px: float | None = None
    patient_age_group: float | None = Field(
        default=None, description="0=Pediatric, 1=Adult, 2=Elderly"
    )
    patient_sex: float | None = Field(
        default=None, description="0=Female (F), 1=Male (M)"
    )


@app.get("/")
def root():
    return {
        "message": "Blood Cell Anomaly Detection API",
        "docs": "/docs",
        "n_features": N_FEATURES,
        "feature_order": FEATURE_ORDER,
        "encoding": {
            "patient_sex": {"F": 0, "M": 1},
            "patient_age_group": {"Pediatric": 0, "Adult": 1, "Elderly": 2},
        },
    }


@app.get("/features")
def list_features():
    return {"n_features": N_FEATURES, "feature_order": FEATURE_ORDER}


@app.post("/predict")
def predict_anomaly(sample: BloodSample):
    try:
        if sample.features is not None:
            if len(sample.features) != N_FEATURES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Expected {N_FEATURES} features, got {len(sample.features)}. "
                        f"Order: {FEATURE_ORDER}"
                    ),
                )
            values = sample.features
        else:
            data = sample.model_dump(exclude={"features"})
            missing = [f for f in FEATURE_ORDER if data.get(f) is None]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing features: {missing}",
                )
            values = [data[f] for f in FEATURE_ORDER]

        input_data = np.array([values], dtype=float)
        scaled = scaler.transform(input_data)
        pred = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]
        confidence = float(np.max(proba) * 100)

        return {
            "status": "success",
            "prediction_code": pred,
            "result": "Anomaly Detected" if pred == 1 else "Normal Cell",
            "confidence_pct": round(confidence, 2),
            "probabilities": {
                "normal": round(float(proba[0]), 4),
                "anomaly": round(float(proba[1]), 4),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}