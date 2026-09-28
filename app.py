import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from pathlib import Path

st.set_page_config(
    page_title="Blood Cell Anomaly Detection",
    page_icon="🩸",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    try:
        model = XGBClassifier()
        model.load_model(str(BASE_DIR / "best_model.json"))
        scaler = joblib.load(BASE_DIR / "scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


model, scaler, load_error = load_artifacts()

NORMAL_SAMPLE = {
    "cell_diameter_um": 10.18,
    "nucleus_area_pct": 43.54,
    "chromatin_density": 0.39,
    "cytoplasm_ratio": 0.56,
    "circularity": 0.77,
    "eccentricity": 0.37,
    "granularity_score": 1.88,
    "lobularity_score": 1.77,
    "membrane_smoothness": 0.84,
    "cell_area_px": 336.57,
    "perimeter_px": 64.07,
    "mean_r": 212.12,
    "mean_g": 146.39,
    "mean_b": 168.66,
    "stain_intensity": 0.62,
    "wbc_count_per_ul": 7043.27,
    "rbc_count_millions_per_ul": 4.79,
    "hemoglobin_g_dl": 13.55,
    "hematocrit_pct": 41.02,
    "platelet_count_per_ul": 249792.62,
    "mcv_fl": 88.94,
    "mchc_g_dl": 33.50,
    "magnification_x": 76.01,
    "image_resolution_px": 336.24,
    "patient_age_group": 1,
    "patient_sex": 0,
}

ANOMALY_SAMPLE = {
    "cell_diameter_um": 15.18,
    "nucleus_area_pct": 58.8,
    "chromatin_density": 0.542,
    "cytoplasm_ratio": 0.301,
    "circularity": 0.563,
    "eccentricity": 0.529,
    "granularity_score": 4.11,
    "lobularity_score": 6.6,
    "membrane_smoothness": 0.80,
    "cell_area_px": 445.0,
    "perimeter_px": 90.0,
    "mean_r": 215.0,
    "mean_g": 141.0,
    "mean_b": 160.0,
    "stain_intensity": 0.555,
    "wbc_count_per_ul": 6352.0,
    "rbc_count_millions_per_ul": 4.44,
    "hemoglobin_g_dl": 11.7,
    "hematocrit_pct": 43.4,
    "platelet_count_per_ul": 257383.0,
    "mcv_fl": 85.5,
    "mchc_g_dl": 31.4,
    "magnification_x": 100.0,
    "image_resolution_px": 224.0,
    "patient_age_group": 2,
    "patient_sex": 0,
}


def _val(key, default):
    return st.session_state.get(key, default)


st.sidebar.title("🩸 Navigation")
app_mode = st.sidebar.radio(
    "Select Section:",
    ["Live Prediction", "Model Comparison", "Visualizations", "About"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Encoding")
st.sidebar.caption("Sex: F=0 · M=1")
st.sidebar.caption("Age: Pediatric=0 · Adult=1 · Elderly=2")

if load_error:
    st.sidebar.error("Model load error: " + str(load_error))
elif model is None:
    st.sidebar.warning("Model not loaded")
else:
    st.sidebar.success("Model loaded")


if app_mode == "Live Prediction":
    st.title("Live Prediction")
    st.write("Enter cell features or load a preset sample, then predict Normal vs Anomaly.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Load Normal Sample", use_container_width=True):
            for k, v in NORMAL_SAMPLE.items():
                st.session_state[k] = v
            st.rerun()
    with c2:
        if st.button("Load Anomaly Sample", use_container_width=True):
            for k, v in ANOMALY_SAMPLE.items():
                st.session_state[k] = v
            st.rerun()

    with st.expander("Cell Morphology", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            cell_diameter_um = st.number_input("cell_diameter_um", value=float(_val("cell_diameter_um", 10.18)))
            nucleus_area_pct = st.number_input("nucleus_area_pct", value=float(_val("nucleus_area_pct", 43.54)))
            chromatin_density = st.number_input("chromatin_density", value=float(_val("chromatin_density", 0.39)))
            cytoplasm_ratio = st.number_input("cytoplasm_ratio", value=float(_val("cytoplasm_ratio", 0.56)))
        with col2:
            circularity = st.number_input("circularity", value=float(_val("circularity", 0.77)))
            eccentricity = st.number_input("eccentricity", value=float(_val("eccentricity", 0.37)))
            granularity_score = st.number_input("granularity_score", value=float(_val("granularity_score", 1.88)))
            lobularity_score = st.number_input("lobularity_score", value=float(_val("lobularity_score", 1.77)))
        with col3:
            membrane_smoothness = st.number_input("membrane_smoothness", value=float(_val("membrane_smoothness", 0.84)))
            cell_area_px = st.number_input("cell_area_px", value=float(_val("cell_area_px", 336.57)))
            perimeter_px = st.number_input("perimeter_px", value=float(_val("perimeter_px", 64.07)))

    with st.expander("Color / Staining"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mean_r = st.number_input("mean_r", value=float(_val("mean_r", 212.12)))
        with col2:
            mean_g = st.number_input("mean_g", value=float(_val("mean_g", 146.39)))
        with col3:
            mean_b = st.number_input("mean_b", value=float(_val("mean_b", 168.66)))
        with col4:
            stain_intensity = st.number_input("stain_intensity", value=float(_val("stain_intensity", 0.62)))

    with st.expander("Lab Blood Values"):
        col1, col2, col3 = st.columns(3)
        with col1:
            wbc_count_per_ul = st.number_input("wbc_count_per_ul", value=float(_val("wbc_count_per_ul", 7043.27)))
            rbc_count_millions_per_ul = st.number_input(
                "rbc_count_millions_per_ul", value=float(_val("rbc_count_millions_per_ul", 4.79))
            )
            hemoglobin_g_dl = st.number_input("hemoglobin_g_dl", value=float(_val("hemoglobin_g_dl", 13.55)))
        with col2:
            hematocrit_pct = st.number_input("hematocrit_pct", value=float(_val("hematocrit_pct", 41.02)))
            platelet_count_per_ul = st.number_input(
                "platelet_count_per_ul", value=float(_val("platelet_count_per_ul", 249792.62))
            )
            mcv_fl = st.number_input("mcv_fl", value=float(_val("mcv_fl", 88.94)))
        with col3:
            mchc_g_dl = st.number_input("mchc_g_dl", value=float(_val("mchc_g_dl", 33.50)))

    with st.expander("Imaging Settings"):
        col1, col2 = st.columns(2)
        with col1:
            magnification_x = st.number_input("magnification_x", value=float(_val("magnification_x", 76.01)))
        with col2:
            image_resolution_px = st.number_input(
                "image_resolution_px", value=float(_val("image_resolution_px", 336.24))
            )

    with st.expander("Patient"):
        col1, col2 = st.columns(2)
        with col1:
            patient_age_group = st.selectbox(
                "Patient Age Group",
                options=[0, 1, 2],
                format_func=lambda x: {0: "0 - Pediatric", 1: "1 - Adult", 2: "2 - Elderly"}[x],
                index=int(_val("patient_age_group", 1)),
            )
        with col2:
            patient_sex = st.selectbox(
                "Patient Sex",
                options=[0, 1],
                format_func=lambda x: {0: "0 - Female", 1: "1 - Male"}[x],
                index=int(_val("patient_sex", 0)),
            )

    st.markdown("---")
    predict_btn = st.button(
        "Predict Anomaly Status",
        type="primary",
        use_container_width=True,
        disabled=(model is None),
    )

    if predict_btn:
        if model is None or scaler is None:
            st.error("Model or scaler not loaded. Check best_model.json and scaler.pkl.")
        else:
            input_data = np.array(
                [
                    [
                        cell_diameter_um,
                        nucleus_area_pct,
                        chromatin_density,
                        cytoplasm_ratio,
                        circularity,
                        eccentricity,
                        granularity_score,
                        lobularity_score,
                        membrane_smoothness,
                        cell_area_px,
                        perimeter_px,
                        mean_r,
                        mean_g,
                        mean_b,
                        stain_intensity,
                        wbc_count_per_ul,
                        rbc_count_millions_per_ul,
                        hemoglobin_g_dl,
                        hematocrit_pct,
                        platelet_count_per_ul,
                        mcv_fl,
                        mchc_g_dl,
                        magnification_x,
                        image_resolution_px,
                        patient_age_group,
                        patient_sex,
                    ]
                ]
            )
            try:
                scaled = scaler.transform(input_data)
                pred = int(model.predict(scaled)[0])
                proba = model.predict_proba(scaled)[0]
                conf = float(np.max(proba) * 100)
                p_normal = float(proba[0])
                p_anomaly = float(proba[1])

                st.subheader("Prediction Result")
                if pred == 1:
                    st.error("Anomaly Detected in Blood Cell")
                else:
                    st.success("Normal Blood Cell (No Anomaly Detected)")

                st.markdown("**Confidence:** {:.2f}%".format(conf))
                st.progress(min(conf / 100.0, 1.0))

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.metric("P(Normal)", "{:.2f}%".format(p_normal * 100))
                    st.progress(p_normal)
                with rc2:
                    st.metric("P(Anomaly)", "{:.2f}%".format(p_anomaly * 100))
                    st.progress(p_anomaly)

                if conf >= 90:
                    st.caption("Model is highly confident in this prediction.")
                elif conf >= 70:
                    st.caption("Model is moderately confident.")
                else:
                    st.caption("Model confidence is relatively low — treat with caution.")

                st.info("Educational demo only — not a medical diagnostic tool.")
            except Exception as e:
                st.error("Error during prediction: " + str(e))


elif app_mode == "Model Comparison":
    st.title("Model Comparison")
    st.write("Performance of all trained models on the held-out test set.")

    comparison_data = {
        "Model": [
            "Gradient Boosting",
            "XGBoost",
            "Random Forest",
            "SVM",
            "Decision Tree",
            "AdaBoost",
            "KNN",
            "Logistic Regression",
            "Naive Bayes",
        ],
        "Accuracy": [0.9770, 0.9762, 0.9651, 0.9379, 0.9286, 0.9141, 0.9124, 0.7849, 0.7789],
        "Precision": [0.9834, 0.9860, 1.0000, 0.9810, 0.9056, 0.9479, 0.9928, 0.6468, 0.7117],
        "Recall": [0.9441, 0.9388, 0.8910, 0.8218, 0.8670, 0.7739, 0.7314, 0.7207, 0.5186],
        "F1 Score": [0.9634, 0.9619, 0.9423, 0.8944, 0.8859, 0.8521, 0.8423, 0.6818, 0.6000],
    }
    results_df = pd.DataFrame(comparison_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown(
        "Note: Gradient Boosting edged slightly (97.70%). "
        "XGBoost (97.62%) was selected for deployment."
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    models_plot = comparison_data["Model"][:7]
    accs = comparison_data["Accuracy"][:7]
    bars = ax.barh(models_plot[::-1], accs[::-1], color="#0d9488")
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_title("Top Models — Accuracy", fontweight="bold")
    for bar, a in zip(bars, accs[::-1]):
        ax.text(a + 0.005, bar.get_y() + bar.get_height() / 2, "{:.4f}".format(a), va="center", fontsize=9)
    st.pyplot(fig)
    plt.close()


elif app_mode == "Visualizations":
    st.title("Visualizations")

    st.subheader("Model Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    models_v = ["Gradient Boosting", "XGBoost", "Random Forest", "SVM", "Decision Tree", "AdaBoost", "KNN"]
    accuracies = [0.9770, 0.9762, 0.9651, 0.9379, 0.9286, 0.9141, 0.9124]
    bars = ax.bar(models_v, accuracies, color="#0d9488")
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Models Accuracy Comparison", fontweight="bold")
    plt.xticks(rotation=15)
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            "{:.4f}".format(acc),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    st.pyplot(fig)
    plt.close()

    st.subheader("Class Balance (Dataset)")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(["Normal (0)", "Anomaly (1)"], [4000, 1880], color=["#22c55e", "#ef4444"])
    ax2.set_ylabel("Count")
    ax2.set_title("Normal vs Anomaly in dataset")
    for i, v in enumerate([4000, 1880]):
        ax2.text(i, v + 50, str(v), ha="center", fontweight="bold")
    st.pyplot(fig2)
    plt.close()

    st.subheader("Cell Diameter — Sample Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sample = np.random.normal(10.18, 1.5, 1000)
    sns.histplot(sample, kde=True, color="#8b5cf6", ax=ax3)
    ax3.set_title("Cell Diameter (um) — illustrative distribution")
    ax3.set_xlabel("Diameter")
    st.pyplot(fig3)
    plt.close()


elif app_mode == "About":
    st.title("About this Project")
    st.markdown(
        """
### Blood Cell Anomaly Detection

Educational ML project that classifies a blood cell as **Normal (0)** or **Anomaly (1)**
from **26** morphology, color, lab, imaging, and patient features.

| Item | Detail |
|------|--------|
| Dataset | 5,880 samples (4,000 Normal / 1,880 Anomaly) |
| Deployed model | XGBoost (~97.6% test accuracy) |
| Scaling | StandardScaler fit on train only |
| Encoding | Sex: F=0, M=1 · Age: Pediatric=0, Adult=1, Elderly=2 |
| App | Streamlit |
| API | FastAPI (api.py) |

### Important disclaimer

This is **not** a medical device.
Predictions are for learning and demonstration only and must not be used for clinical diagnosis.

### How to run

- Streamlit: `python -m streamlit run app.py`
- API: `uvicorn api:app --reload --port 8000`
"""
    )