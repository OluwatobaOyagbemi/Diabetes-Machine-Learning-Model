# Diabetes Prediction — Machine Learning Project

## Overview
A machine learning project to predict diabetes outcomes using a clinical dataset of **100,000 patient records** and **9 features**, including age, BMI, HbA1c level, blood glucose level, hypertension, heart disease, smoking history, and gender.

---

## Dataset
| Property | Detail |
|---|---|
| Shape | (100,000, 9) |
| Target Variable | `diabetes` (0 = No Diabetes, 1 = Diabetes) |
| Source | Clinical patient records |

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
Used core pandas functions to understand the dataset before any processing:
- `.shape` — confirmed dataset dimensions
- `.head()` — inspected feature structure and values
- `.info()` — checked data types and null values
- `.describe()` — reviewed distribution of numerical features
- `.value_counts()` — examined categorical columns (gender, smoking history)

---

### 2. Correlation Analysis
Examined feature correlations with the diabetes outcome:

| Rank | Feature | Notes |
|---|---|---|
| 1st | `blood_glucose_level` | Most direct real-time indicator of diabetes |
| 2nd | `HbA1c_level` | Clinical biomarker reflecting average blood sugar over 2–3 months |

> **HbA1c** is a recognised clinical biomarker for diabetes diagnosis. Unlike a single blood glucose reading, it provides a longer-term view of a patient's metabolic state, making it a highly informative feature for prediction.

---

### 3. Class Imbalance
The target variable was imbalanced — non-diabetic cases significantly outnumbered diabetic ones.

**Why this matters:** A model trained on imbalanced data tends to predict the majority class and still report high accuracy — making accuracy alone a misleading metric.

**Solution — SMOTE (Synthetic Minority Oversampling Technique):**
Rather than duplicating existing minority records, SMOTE generates synthetic diabetic samples by interpolating between existing data points, producing a more balanced and representative training set.

---

### 4. Model Training & Evaluation
Three models were trained and evaluated:

| Model | Accuracy | F1-Score | ROC-AUC |
| **Random Forest** ✅ | **0.975** | **0.975** | **0.997** |

**Evaluation metrics used:**
- **Precision** — Of all predicted diabetic cases, how many were correct?
- **Recall** — Of all actual diabetic cases, how many were identified?
- **F1-Score** — Balance between precision and recall
- **ROC-AUC** — Overall ability to distinguish between classes

> Accuracy alone was not used as the primary metric due to the initial class imbalance in the dataset.

---

## Libraries Used
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn (SMOTE)
xgboost
```

---

## Results
The **Random Forest** model performed best with:
- Accuracy: **97.5%**
- F1-Score: **0.975**
- ROC-AUC: **0.997**
