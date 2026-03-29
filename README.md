#  Hospital Readmission Risk Prediction

##  Overview

This project predicts whether a patient is likely to be readmitted within 30 days using machine learning.

It helps healthcare providers identify high-risk patients and take preventive actions.

---

##  Objective

* Predict 30-day hospital readmission
* Handle imbalanced healthcare data
* Provide risk-based patient categorization

---

##  Dataset

* 100K+ hospital records
* Features include:

  * Patient demographics
  * Admission details
  * Lab procedures
  * Medical history

---

##  Approach

### 1. Data Cleaning

* Handled missing values (`? --> NaN`)
* Removed noisy and irrelevant columns
* Standardized categorical variables

### 2. Feature Engineering

* Converted ID columns to categorical
* Separated numerical and categorical features

### 3. Modeling

* Logistic Regression (selected model)
* Random Forest (baseline comparison)

### 4. Evaluation

* ROC-AUC
* PR-AUC
* Recall-focused evaluation (healthcare priority)

---

##  Key Insights

* Patients with more inpatient visits → higher risk
* Longer hospital stays → higher readmission probability
* Older age groups → slightly higher risk

---

##  Model Performance

| Model               | Recall | ROC-AUC |
| ------------------- | ------ | ------- |
| Logistic Regression | ~58%   | ~0.67   |
| Random Forest       | ~1%    | ~0.67   |

 Logistic Regression selected (better recall)

---

##  Threshold Strategy

* Lower threshold → higher recall
* Used for early detection of high-risk patients

---

##  Explainability (SHAP)

* Identifies key drivers of readmission
* Improves trust in model predictions

---

##  How to Use

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run prediction

```bash
python src/predict.py
```

### Step 3: Input your dataset

* Must contain similar columns as training data
* Missing columns will be auto-handled

---

##  Output

The model generates:

* `prediction` → 0 or 1
* `probability` → risk score
* `risk_level` → Low / Medium / High

---

##  Limitations

* Not clinically validated
* Moderate precision (false positives exist)
* Depends on data quality

---

##  Future Improvements

* SMOTE / advanced imbalance handling
* XGBoost / LightGBM
* Model explainability dashboard
* Deployment as API

---

##  Author

Data Analyst | Machine Learning Enthusiast

---
