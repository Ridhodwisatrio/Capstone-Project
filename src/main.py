from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import os

app = FastAPI(title="VitalsCheck API")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "model")

diseases = ['Diabetes', 'Penyakit Jantung', 'Stroke', 'Hipertensi', 'Kolesterol Tinggi']

diseases_thresholds = {
    'Diabetes'          : 0.42,
    'Penyakit Jantung'  : 0.46,
    'Stroke'            : 0.45,
    'Hipertensi'        : 0.52,
    'Kolesterol Tinggi' : 0.50
}

base_features = [
    'Age', 'Sex', 'BMI', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'CholCheck', 'Smoker', 'PhysActivity',
    'Fruits', 'Veggies', 'HvyAlcoholConsump'
]

loaded_models = {}
loaded_scalers = {}

# Load model dan scaler
@app.on_event("startup")
def load_all_models_and_scalers():
        file_mapping = {
            'Diabetes': ('Diabetes_binary_model.keras', 'scaler_Diabetes_binary.pkl'),
            'Penyakit Jantung': ('HeartDiseaseor_model.keras', 'scaler_HeartDiseaseorAttack.pkl'),
            'Stroke': ('Stroke_model.keras', 'scaler_Stroke.pkl'),
            'Hipertensi': ('HighBP_model.keras', 'scaler_HighBP.pkl'),
            'Kolesterol Tinggi': ('HighChol_model.keras', 'scaler_HighChol.pkl')
        }
        
        for disease, (m_file, s_file) in file_mapping.items():
            model_path = os.path.join(model_dir, m_file)
            scaler_path = os.path.join(model_dir, s_file)
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                loaded_models[disease] = tf.keras.models.load_model(model_path)
                loaded_scalers[disease] = joblib.load(scaler_path)
                print(f"-> Sukses load model & scaler: {disease}")
            else:
                print(f"File tidak ditemukan untuk {disease}: {model_path} / {scaler_path}")

class PatientFeatures(BaseModel):
    Age: float
    Sex: float
    BMI: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    CholCheck: float
    Smoker: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float

class DiseasePredictionDetail(BaseModel):
    probability_score: float
    threshold_used: float
    predicted_risk: str

class MultiPredictionResponse(BaseModel):
    status: str
    predictions: dict[str, DiseasePredictionDetail]

# 3. Endpoints API
@app.get("/")
def root():
    return {
        "message": "Welcome to VitalsCheck API",
        "available_diseases": diseases,
        "thresholds": diseases_thresholds
    }

@app.post("/predict", response_model=MultiPredictionResponse)
def predict_all_diseases(features: PatientFeatures):
    if not loaded_models:
        raise HTTPException(status_code=500, detail="Model belum ter-load di server.")
    
    try:
        raw_data_dict = {
            'Age': [features.Age], 'Sex': [features.Sex], 'BMI': [features.BMI], 
            'GenHlth': [features.GenHlth], 'MentHlth': [features.MentHlth], 
            'PhysHlth': [features.PhysHlth], 'DiffWalk': [features.DiffWalk], 
            'CholCheck': [features.CholCheck], 'Smoker': [features.Smoker], 
            'PhysActivity': [features.PhysActivity], 'Fruits': [features.Fruits], 
            'Veggies': [features.Veggies], 'HvyAlcoholConsump': [features.HvyAlcoholConsump]
        }
        
        df_input = pd.DataFrame(raw_data_dict)[base_features]
        results = {}
        
        for nama_penyakit in diseases:
            if nama_penyakit not in loaded_models:
                continue
                
            df_temp = df_input.copy()
            
            if nama_penyakit in ['Diabetes', 'Penyakit Jantung', 'Stroke']:
                df_temp['BMI_Smoker'] = df_temp['BMI'] * df_temp['Smoker']
                df_temp['Unhealthy_Index'] = df_temp['BMI'] * (1 - df_temp['PhysActivity'])
                df_temp['Age_GenHlth'] = df_temp['Age'] * df_temp['GenHlth']
                fitur_lengkap = base_features + ['BMI_Smoker', 'Unhealthy_Index', 'Age_GenHlth']
                
            else: # Kelompok hipertensi dan kolesterol
                df_temp['Age_CholCheck'] = df_temp['Age'] * df_temp['CholCheck']
                df_temp['Stress_Physical_Index'] = df_temp['MentHlth'] * df_temp['PhysHlth']
                df_temp['BMI_Alcohol'] = df_temp['BMI'] * df_temp['HvyAlcoholConsump']
                fitur_lengkap = base_features + ['Age_CholCheck', 'Stress_Physical_Index', 'BMI_Alcohol']
            
            input_data = df_temp[fitur_lengkap].values
            
            input_scaled = loaded_scalers[nama_penyakit].transform(input_data)
            input_tensor = tf.convert_to_tensor(input_scaled, dtype=tf.float32)
            
            prediction = loaded_models[nama_penyakit](input_tensor, training=False)
            probabilitas = float(prediction[0][0])
            
            status_risiko = "Tinggi" if probabilitas > diseases_thresholds[nama_penyakit] else "Rendah"
            
            results[nama_penyakit] = DiseasePredictionDetail(
                probability_score=round(probabilitas, 4),
                threshold_used=diseases_thresholds[nama_penyakit],
                predicted_risk=status_risiko
            )
            
        return MultiPredictionResponse(status="success", predictions=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat inference: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "total_models_loaded": len(loaded_models), 
        "loaded_diseases": list(loaded_models.keys()) 
    }