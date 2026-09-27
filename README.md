# 🩸 Blood Cell Anomaly Detection App

**Machine Learning Classification + Streamlit Web App + FastAPI**

توقع حالة خلية الدم (طبيعية أو غير طبيعية) من 26 خاصية بيولوجية ومخبرية، مع نوتبوك كامل يوثّق التحليل والتدريب ومقارنة 9 موديلات.

---

## 📋 نظرة عامة

| القيمة | التصنيف              |
|--------|----------------------|
| `0`    | ✅ Normal            |
| `1`    | ⚠️ Abnormal / Anomaly |

### 🔐 الترميز المستخدم

| الخاصية              | الترميز                                              |
|----------------------|------------------------------------------------------|
| `patient_sex`        | 👩 **F = 0** &nbsp;|&nbsp; 👨 **M = 1**              |
| `patient_age_group`  | 👶 Pediatric=0 &nbsp;|&nbsp; 🧑 Adult=1 &nbsp;|&nbsp; 👴 Elderly=2 |

---

## 📁 هيكل المشروع

```
.
├── 🖥️  app.py                              # تطبيق Streamlit
├── 🔌  api.py                              # خدمة FastAPI
├── 📓  Blood_Cell_Anomaly_Detection.ipynb  # النوتبوك الكامل
├── 🤖  best_model.json                     # الموديل المنشور (XGBoost)
├── 🤖  best_model.pkl                      # نسخة joblib (تحتاج نفس إصدار XGBoost)
├── 📏  scaler.pkl                          # StandardScaler
├── 📋  feature_order.pkl                   # ترتيب الـ 26 feature
├── 📊  blood_cell_anomaly_detection.csv    # الداتا
├── 📦  requirements.txt
└── 📄  README.md
```

---

## 🧬 الفيتشرز المستخدمة (FEATURE_ORDER)

**26 خاصية** — الترتيب ثابت ولازم يتبع في أي استدعاء:

| # | Feature | # | Feature |
|---|---------|---|---------|
| 1 | `cell_diameter_um` | 14 | `mean_b` |
| 2 | `nucleus_area_pct` | 15 | `stain_intensity` |
| 3 | `chromatin_density` | 16 | `wbc_count_per_ul` |
| 4 | `cytoplasm_ratio` | 17 | `rbc_count_millions_per_ul` |
| 5 | `circularity` | 18 | `hemoglobin_g_dl` |
| 6 | `eccentricity` | 19 | `hematocrit_pct` |
| 7 | `granularity_score` | 20 | `platelet_count_per_ul` |
| 8 | `lobularity_score` | 21 | `mcv_fl` |
| 9 | `membrane_smoothness` | 22 | `mchc_g_dl` |
| 10 | `cell_area_px` | 23 | `magnification_x` |
| 11 | `perimeter_px` | 24 | `image_resolution_px` |
| 12 | `mean_r` | 25 | `patient_age_group` |
| 13 | `mean_g` | 26 | `patient_sex` |

> ⛔ **تم استبعادها** (لتجنب data leakage / metadata):  
> `cell_id` · `disease_category` · `cell_type` · `dataset_source` · `staining_protocol` · `microscope_model` · `cytodiffusion_anomaly_score` · `cytodiffusion_classification_confidence` · `labeller_confidence_score`

---

## 🏆 مقارنة الموديلات (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| 🥇 Gradient Boosting | 0.9770 | 0.9834 | 0.9441 | 0.9634 |
| ⭐ **XGBoost** (Deployed) | **0.9762** | 0.9860 | 0.9388 | 0.9619 |
| 🥉 Random Forest | 0.9651 | 1.0000 | 0.8910 | 0.9423 |
| SVM | 0.9379 | 0.9810 | 0.8218 | 0.8944 |
| Decision Tree | 0.9286 | 0.9056 | 0.8670 | 0.8859 |
| AdaBoost | 0.9141 | 0.9479 | 0.7739 | 0.8521 |
| KNN | 0.9124 | 0.9928 | 0.7314 | 0.8423 |
| Logistic Regression | 0.7849 | 0.6468 | 0.7207 | 0.6818 |
| Naive Bayes | 0.7789 | 0.7117 | 0.5186 | 0.6000 |

> 📌 الموديل المنشور: **XGBoost** عبر `best_model.json`  
> ⚠️ النتائج خاصة بهذه الداتا والـ split — **ليست أداة تشخيص طبي**.

---

## 🖥️ تطبيق Streamlit

ثلاثة أقسام من الـ Sidebar:

| القسم | الوظيفة |
|-------|---------|
| 🔮 **Live Prediction** | إدخال الـ 26 feature → نتيجة + نسبة ثقة |
| 📊 **Model Comparison** | جدول مقارنة الـ 9 موديلات |
| 📈 **Visualizations** | رسم Accuracy + توزيع Cell Diameter |

```bash
python -m streamlit run app.py
```
يفتح على: `http://localhost:8501`

---

## 🔌 FastAPI

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

| Endpoint | الوصف |
|----------|--------|
| `GET /` | معلومات الخدمة + ترتيب الفيتشرز + الترميز |
| `GET /features` | قائمة الـ 26 feature |
| `POST /predict` | التوقع (قائمة أو حقول بأسماء) |

📖 Docs التفاعلية: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**مثال طلب:**
```json
{
  "features": [
    10.18, 43.54, 0.39, 0.56, 0.77, 0.37, 1.88, 1.77, 0.84,
    336.57, 64.07, 212.12, 146.39, 168.66, 0.62,
    7043.27, 4.79, 13.55, 41.02, 249792.62, 88.94, 33.50,
    76.01, 336.24, 1, 0
  ]
}
```
*(آخر رقمين: age=1 Adult · sex=0 Female)*

---

## ⚙️ التثبيت والتشغيل

```bash
# 1) ادخل فولدر المشروع
cd BloodCellAnomalyDetection

# 2) بيئة افتراضية (مستحسن)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 3) المكتبات
pip install -r requirements.txt
pip install xgboost==3.4.1     # نفس إصدار Colab

# 4) Streamlit
python -m streamlit run app.py

# 5) API
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

✅ تأكد إن الملفات دي جنب `app.py` و `api.py`:
- `best_model.json`
- `scaler.pkl`
- `feature_order.pkl`

---

## 🛠️ Tech Stack

| التقنية | الاستخدام |
|---------|-----------|
| Python 3 | اللغة الأساسية |
| Pandas / NumPy | معالجة البيانات |
| Scikit-learn | 8 موديلات تصنيف |
| **XGBoost** | الموديل المنشور |
| Matplotlib / Seaborn | الرسوم والتحليل |
| Streamlit | واجهة المستخدم |
| FastAPI + Uvicorn | الـ API |
| Joblib | حفظ/تحميل الـ scaler |

---

## 📌 ملاحظات مهمة

- ✅ الـ outliers اتفحصت وماتشالتش (ممكن تكون قيم طبية حقيقية)
- ✅ `StandardScaler` اتعمل fit على **training data فقط**
- ✅ الـ train/test split كان **stratified**
- ✅ لازم نفس الترميز ونفس `FEATURE_ORDER` في أي inference
- ✅ استخدم `best_model.json` بدل `.pkl` لتجنب تعارض إصدارات XGBoost

---

## 📄 الترخيص

مشروع تعليمي / تجريبي فقط.  
**ليس أداة تشخيص طبي.**
