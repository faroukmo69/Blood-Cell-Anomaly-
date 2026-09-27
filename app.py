import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# إعدادات الصفحة
st.set_page_config(
    page_title="Blood Cell Anomaly Detection",
    layout="wide"
)

# تحميل الموديل والـ Scaler من JSON (أضمن من pkl)
@st.cache_resource
def load_artifacts():
    try:
        model = XGBClassifier()
        model.load_model("best_model.json")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.sidebar.error(f"Load error: {e}")
        return None, None

model, scaler = load_artifacts()

# عنوان التطبيق
st.title("Blood Cell Anomaly Detection App")
st.write("Enter the biological features in the sidebar to predict whether the blood cell is Normal or Anomaly.")

# الشريط الجانبي للتنقل
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Section:", ["Live Prediction", "Model Comparison", "Visualizations"])

if app_mode == "Live Prediction":
    st.sidebar.header("Input Cell Features")
    
    # المدخلات الـ 26 بنفس ترتيب FEATURE_ORDER
    cell_diameter_um = st.sidebar.number_input("cell_diameter_um", value=10.18)
    nucleus_area_pct = st.sidebar.number_input("nucleus_area_pct", value=43.54)
    chromatin_density = st.sidebar.number_input("chromatin_density", value=0.39)
    cytoplasm_ratio = st.sidebar.number_input("cytoplasm_ratio", value=0.56)
    circularity = st.sidebar.number_input("circularity", value=0.77)
    eccentricity = st.sidebar.number_input("eccentricity", value=0.37)
    granularity_score = st.sidebar.number_input("granularity_score", value=1.88)
    lobularity_score = st.sidebar.number_input("lobularity_score", value=1.77)
    membrane_smoothness = st.sidebar.number_input("membrane_smoothness", value=0.84)
    cell_area_px = st.sidebar.number_input("cell_area_px", value=336.57)
    perimeter_px = st.sidebar.number_input("perimeter_px", value=64.07)
    mean_r = st.sidebar.number_input("mean_r", value=212.12)
    mean_g = st.sidebar.number_input("mean_g", value=146.39)
    mean_b = st.sidebar.number_input("mean_b", value=168.66)
    stain_intensity = st.sidebar.number_input("stain_intensity", value=0.62)
    wbc_count_per_ul = st.sidebar.number_input("wbc_count_per_ul", value=7043.27)
    rbc_count_millions_per_ul = st.sidebar.number_input("rbc_count_millions_per_ul", value=4.79)
    hemoglobin_g_dl = st.sidebar.number_input("hemoglobin_g_dl", value=13.55)
    hematocrit_pct = st.sidebar.number_input("hematocrit_pct", value=41.02)
    platelet_count_per_ul = st.sidebar.number_input("platelet_count_per_ul", value=249792.62)
    mcv_fl = st.sidebar.number_input("mcv_fl", value=88.94)
    mchc_g_dl = st.sidebar.number_input("mchc_g_dl", value=33.50)
    magnification_x = st.sidebar.number_input("magnification_x", value=76.01)
    image_resolution_px = st.sidebar.number_input("image_resolution_px", value=336.24)
    
    patient_age_group = st.sidebar.selectbox(
        "Patient Age Group (0: Pediatric, 1: Adult, 2: Elderly)", 
        [0, 1, 2], 
        index=1
    )
    patient_sex = st.sidebar.selectbox(
        "Patient Sex (0: Female, 1: Male)", 
        [0, 1], 
        index=0
    )
    
    predict_button = st.sidebar.button("Predict Anomaly Status")
    st.subheader("Prediction Result:")
    
    if predict_button:
        if model is not None and scaler is not None:
            # ترتيب الـ 26 feature مطابق لـ FEATURE_ORDER
            input_data = np.array([[
                cell_diameter_um, nucleus_area_pct, chromatin_density, cytoplasm_ratio,
                circularity, eccentricity, granularity_score, lobularity_score,
                membrane_smoothness, cell_area_px, perimeter_px, mean_r, mean_g, mean_b,
                stain_intensity, wbc_count_per_ul, rbc_count_millions_per_ul, hemoglobin_g_dl,
                hematocrit_pct, platelet_count_per_ul, mcv_fl, mchc_g_dl, magnification_x,
                image_resolution_px, patient_age_group, patient_sex
            ]])
            
            try:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                proba = model.predict_proba(input_scaled)
                confidence = float(np.max(proba) * 100)

                if prediction[0] == 1:
                    st.error("⚠️ Anomaly Detected in Blood Cell")
                else:
                    st.success("✅ Normal Blood Cell (No Anomaly Detected)")
                    
                st.info(f"Model Confidence Level: {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning(
                "Model or Scaler not found. Make sure best_model.json and scaler.pkl "
                "are in the same folder."
            )

elif app_mode == "Model Comparison":
    st.subheader("Models Comparison Dashboard")
    st.markdown("This table summarizes the performance of all trained classification models on the test set:")
    
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
        "Accuracy":  [0.9770, 0.9762, 0.9651, 0.9379, 0.9286, 0.9141, 0.9124, 0.7849, 0.7789],
        "Precision": [0.9834, 0.9860, 1.0000, 0.9810, 0.9056, 0.9479, 0.9928, 0.6468, 0.7117],
        "Recall":    [0.9441, 0.9388, 0.8910, 0.8218, 0.8670, 0.7739, 0.7314, 0.7207, 0.5186],
        "F1 Score":  [0.9634, 0.9619, 0.9423, 0.8944, 0.8859, 0.8521, 0.8423, 0.6818, 0.6000],
    }
    
    results_df = pd.DataFrame(comparison_data)
    st.dataframe(results_df, use_container_width=True)
    st.markdown(
        "**Note:** Gradient Boosting edged slightly (97.70%). "
        "**XGBoost** (97.62%) was selected for deployment."
    )

elif app_mode == "Visualizations":
    st.subheader("Exploratory Data Analysis & Visualizations")
    st.markdown("Here are some key visualizations representing the blood cell dataset and model performance.")
    
    st.markdown("### Model Accuracy Comparison")
    fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
    models = [
        "Gradient Boosting", "XGBoost", "Random Forest",
        "SVM", "Decision Tree", "AdaBoost", "KNN"
    ]
    accuracies = [0.9770, 0.9762, 0.9651, 0.9379, 0.9286, 0.9141, 0.9124]
    
    bars = ax_acc.bar(models, accuracies, color='teal')
    ax_acc.set_ylim(0.80, 1.0)
    ax_acc.set_title("Models Accuracy Comparison", fontsize=14, fontweight='bold')
    ax_acc.set_ylabel("Accuracy")
    plt.xticks(rotation=20)
    
    for bar, acc in zip(bars, accuracies):
        ax_acc.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{acc:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    st.pyplot(fig_acc)
    
    st.markdown("### Feature Distribution Sample")
    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sample_data = np.random.normal(10.18, 1.5, 1000)
    sns.histplot(sample_data, kde=True, color='purple', ax=ax_dist)
    ax_dist.set_title("Cell Diameter Distribution (um)")
    ax_dist.set_xlabel("Diameter")
    ax_dist.set_ylabel("Count")
    st.pyplot(fig_dist)