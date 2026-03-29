#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# PATIENT READMISSION PREDICTION

import pandas as pd
import joblib
import os

pd.set_option('future.no_silent_downcasting', True)

# LOAD MODEL + METADATA


MODEL_PATH = "production_model.pkl"
COLUMNS_PATH = "model_columns.pkl"
DTYPES_PATH = "model_dtypes.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

if not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError("Model columns file not found")

if not os.path.exists(DTYPES_PATH):
    raise FileNotFoundError("Model dtypes file not found")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)
model_dtypes = joblib.load(DTYPES_PATH)

# VALIDATION

def validate_input(df):
    if df.empty:
        raise ValueError("Input data is empty")
    
    if df.shape[1] < 5:
        raise ValueError("Invalid dataset: too few columns")
    
    return df

# CLEANING FUNCTION

def clean_data(df):
    df = df.copy()
    
    df.replace('?', pd.NA, inplace=True)
    
    df.drop(columns=['weight','payer_code'], errors='ignore', inplace=True)
    df.drop(columns=['max_glu_serum','A1Cresult'], errors='ignore', inplace=True)
    
    for col in ['race','medical_specialty']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    df.drop(columns=['encounter_id','patient_nbr'], errors='ignore', inplace=True)
    
    return df

# VALUE FIXES 

def convert_age(age):
    if isinstance(age, str):
        return age
    try:
        age = int(age)
        bins = [(0,10),(10,20),(20,30),(30,40),(40,50),
                (50,60),(60,70),(70,80),(80,90),(90,100)]
        for low, high in bins:
            if low <= age < high:
                return f"[{low}-{high})"
    except:
        return "Unknown"

def cap_values(df):
    if 'time_in_hospital' in df.columns:
        df['time_in_hospital'] = pd.to_numeric(
            df['time_in_hospital'], errors='coerce'
        )
        df['time_in_hospital'] = df['time_in_hospital'].fillna(0)
        df['time_in_hospital'] = df['time_in_hospital'].clip(0, 30)
    return df

# ALIGN + TYPE FIX FUNCTION

def align_input_data(df, reference_columns, reference_dtypes):
    
    df = df.copy()
    
    for col in reference_columns:
        if col not in df.columns:
            if str(reference_dtypes[col]) == 'object':
                df[col] = "Unknown"
            else:
                df[col] = 0
    
    df = df[reference_columns]
    
    for col in df.columns:
        try:
            df[col] = df[col].astype(reference_dtypes[col])
        except:
            pass
    
    return df

# RISK FUNCTION

def categorize_risk(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

# MAIN PREDICTION FUNCTION

def predict(file_path, output_path="predictions.csv"):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError("Input file not found")
    
    print(" Loading data...")
    df = pd.read_csv(file_path)
    
    df = validate_input(df)
    
    print(" Cleaning data...")
    df = clean_data(df)
    
    df = cap_values(df)
    
    # Convert age 
    if 'age' in df.columns:
        df['age'] = df['age'].apply(convert_age)
    
    X = df.drop('readmitted', axis=1, errors='ignore')
    
    print(" Aligning columns...")
    X = align_input_data(X, model_columns, model_dtypes)
    
    # Handle missing values
    for col in X.columns:
        if str(model_dtypes[col]) == 'object':
            X[col] = X[col].fillna("Unknown").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(" Making predictions...")
    df['prediction'] = model.predict(X)
    df['probability'] = model.predict_proba(X)[:,1]
    
    df['risk_level'] = df['probability'].apply(categorize_risk)
    
    df.to_csv(output_path, index=False)
    
    print(f" Predictions saved to {output_path}")

# RUN SCRIPT

if __name__ == "__main__":
    
    input_file = input("Enter file path: ")
    
    predict(input_file)


# In[ ]:




