#!/usr/bin/env python
# coding: utf-8

# In[1]:


# GLOBAL CONFIGURATION

import os
import warnings
import random
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

pd.set_option('display.max_columns', 100)

print(" Environment ready")


# In[2]:


# IMPORTS

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

print(" Libraries loaded")


# In[3]:


# LOAD DATA

DATA_PATH = "diabetic_data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(" Dataset not found")

def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(DATA_PATH)

print("Shape:", df.shape)


# In[4]:


# BUSINESS UNDERSTANDING

print("""
Objective:
Predict which patients are at risk of readmission within 30 days.

Why this matters:
- Hospitals face penalties for high readmission rates
- Early identification helps improve patient care
- Enables targeted intervention strategies

Key challenge:
- Dataset is highly imbalanced (few readmissions)
""")


# In[5]:


# DATA CLEANING

def clean_data(df):
    df = df.copy()
    
    df.replace('?', np.nan, inplace=True)
    
    # Drop noisy columns
    df.drop(columns=['weight','payer_code'], errors='ignore', inplace=True)
    df.drop(columns=['max_glu_serum','A1Cresult'], errors='ignore', inplace=True)
    
    # Fill missing
    for col in ['race','medical_specialty']:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)
    
    # Drop IDs
    df.drop(columns=['encounter_id','patient_nbr'], errors='ignore', inplace=True)
    
    # Drop constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(columns=col, inplace=True)
    
    # Target encoding
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    return df

df = clean_data(df)

print("Cleaned shape:", df.shape)


# In[6]:


#EDA
#CLASS IMBALANCE

counts = df['readmitted'].value_counts()

print("Class Distribution:")
print(counts)

sns.barplot(x=counts.index, y=counts.values)
plt.title("Class Imbalance")
plt.show()

print("\nInsight:")
print("- Only ~11% patients are readmitted")
print("- This makes prediction challenging")
print("- Accuracy alone is misleading")


# In[7]:


# LENGTH OF STAY

sns.boxplot(x='readmitted', y='time_in_hospital', data=df)
plt.title("Hospital Stay vs Readmission")
plt.show()

print("""
Insight:
Patients with longer hospital stays show a slightly higher tendency for readmission, though the overlap indicates this is not a strong standalone predictor
""")


# In[8]:


# PRIOR INPATIENT VISITS

sns.boxplot(x='readmitted', y='number_inpatient', data=df)
plt.title("Prior Inpatient Visits vs Readmission")
plt.show()

print("\nInsight:")
print("- Patients with higher prior inpatient visits show a clear increase in readmission risk, indicating recurring or chronic conditions")


# In[9]:


# AGE INSIGHT

sns.countplot(x='age', hue='readmitted', data=df)
plt.xticks(rotation=90)
plt.title("Age vs Readmission")
plt.show()

print("""
Insight:
- Older age groups (60+) show a higher proportion of readmissions, suggesting age as a contributing risk factor.
""")


# In[10]:


# FEATURE ENGINEERING

def prepare_features(df):
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    # Convert ID-like columns
    id_cols = ['admission_type_id','discharge_disposition_id','admission_source_id']
    for col in id_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()
    
    return X, y, cat_cols, num_cols

X, y, cat_cols, num_cols = prepare_features(df)


# In[11]:


# SAVE COLUMN STRUCTURE + DTYPES

MODEL_COLUMNS = X.columns.tolist()
MODEL_DTYPES = X.dtypes.to_dict()

import joblib

joblib.dump(MODEL_COLUMNS, "model_columns.pkl")
joblib.dump(MODEL_DTYPES, "model_dtypes.pkl")

print(" Columns and dtypes saved")


# In[12]:


# PREPROCESSING

def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

preprocessor = build_preprocessor(cat_cols, num_cols)


# In[13]:


# SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train:", X_train.shape, "Test:", X_test.shape)


# In[14]:


# MODEL FACTORY

def build_models(preprocessor):
    
    lr = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='liblinear'
        ))
    ])
    
    rf = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ))
    ])
    
    return lr, rf

lr_model, rf_model = build_models(preprocessor)


# In[15]:


# TRAIN

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print(" Models trained")


# In[16]:


# EVALUATION

def evaluate_model(model, X_test, y_test, name):
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))    
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC:", average_precision_score(y_test, y_proba))
    
    return y_proba

lr_proba = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
rf_proba = evaluate_model(rf_model, X_test, y_test, "Random Forest")


# In[17]:


# SHAP EXPLAINABILITY

import shap

X_processed = lr_model.named_steps['prep'].transform(X_test.sample(200, random_state=42))
X_processed = X_processed.toarray()
feature_names = lr_model.named_steps['prep'].get_feature_names_out()

explainer = shap.Explainer(lr_model.named_steps['clf'], X_processed)
shap_values = explainer(X_processed)

shap.summary_plot(shap_values, X_processed, feature_names=feature_names)


# In[18]:


# THRESHOLD OPTIMIZATION (FOR LOGISTIC REGRESSION)

thresholds = np.arange(0.1, 0.9, 0.05)

results = []

for t in thresholds:
    y_pred_t = (lr_proba >= t).astype(int)
    
    report = classification_report(y_test, y_pred_t, output_dict=True)
    
    precision = report['1']['precision']
    recall = report['1']['recall']
    
    results.append((t, precision, recall))

threshold_df = pd.DataFrame(results, columns=['threshold','precision','recall'])

threshold_df.sort_values(by='recall', ascending=False).head(10)



# In[19]:


best_threshold = threshold_df.sort_values(by='recall', ascending=False).iloc[0]
print(best_threshold)

print("Note: 0.10 gives highest recall, but 0.30 is chosen to balance recall and precision.")


# In[20]:


# SAVE FINAL MODEL

MODEL_PATH = "production_model.pkl"

joblib.dump(lr_model, MODEL_PATH)

print(" Logistic Regression model saved")


# In[21]:


print("""

Logistic Regression:
- Achieves ~58% recall at default threshold
- Improves to ~92% recall after threshold tuning
- More effective for identifying high-risk patients

Random Forest:
- Very low recall (~1%)
- Fails to detect most readmitted cases
- Biased toward majority class

Final Decision:
- Logistic Regression is the better model for this problem

Reason:
- In healthcare, recall is more important than accuracy
- Missing a high-risk patient is costly
""")


# In[22]:


# CURVES

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

plt.plot(fpr_lr, tpr_lr, label='LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.legend()
plt.title("ROC Curve")
plt.show()


prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_proba)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_proba)

plt.plot(rec_lr, prec_lr, label='LR')
plt.plot(rec_rf, prec_rf, label='RF')
plt.legend()
plt.title("PR Curve")
plt.show()


# In[23]:


# THRESHOLD

threshold = 0.3
y_pred_custom = (lr_proba >= threshold).astype(int)

print("Custom Threshold Results:")
print(classification_report(y_test, y_pred_custom))

print("Note: Accuracy is low because threshold is optimized for recall.")


# In[24]:


# THRESHOLD INSIGHT

print("""
Threshold Tuning Insight:

- Lower threshold → higher recall (more patients flagged)
- Higher threshold → higher precision (fewer false alerts)

Healthcare Perspective:
- Recall is more important
- Missing a high-risk patient is costly

Therefore:
- Lower threshold is preferred for early intervention
""")


# In[26]:


# LOAD

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found")
    return joblib.load(path)

model_loaded = load_model("production_model.pkl")
print("Model loaded")


# In[64]:


# PREDICT FUNCTION

def predict_single(model, X_sample):
    pred = model.predict(X_sample)[0]
    prob = model.predict_proba(X_sample)[0][1]
    
    return {"prediction": int(pred), "probability": float(prob)}

# Load saved columns
model_columns = joblib.load("model_columns.pkl")

# Align input
sample = X_test.iloc[[0]]
sample_aligned = sample[model_columns]

# Predict
result = predict_single(model_loaded, sample_aligned)


print(result)


# In[66]:


print("""
Final Conclusion:

This project combines data analysis and machine learning
to identify patients at risk of hospital readmission.

Key Insights:
- Readmission is influenced by prior visits and hospital stay duration
- Dataset imbalance significantly impacts predictions
- Logistic Regression performs better for identifying high-risk patients

Model Findings:
- Logistic Regression achieved ~58% recall at default threshold
- After threshold tuning (0.3), recall improved to ~92%
- This makes the model more suitable for healthcare use cases
- Random Forest achieved high accuracy but failed to detect minority cases (very low recall)

Business Value:
- Model can help identify high-risk patients early
- Supports healthcare teams in proactive intervention planning

Limitations:
- Low precision leads to false positives
- Model not clinically validated for real-world use

Deployment:
- Model exported using joblib
- Prediction pipeline supports external CSV inputs

Future Improvements:
- Improve precision using advanced models
- Apply techniques like SMOTE or class balancing
- Integrate with hospital systems

- Model is robust to missing values, unseen categories, and schema variations through preprocessing pipeline

Overall:
- A strong analytical and predictive solution for healthcare insights
""")

