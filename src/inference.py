import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import os

# Variable untuk 13 fitur
AGE = 9.0
SEX = 1.0
BMI = 28.3
GEN_HLTH = 3.0
MENT_HLTH = 0.0
PHYS_HLTH = 2.0
DIFF_WALK = 0.0
CHOL_CHECK = 1.0
SMOKER = 1.0
PHYS_ACTIVITY = 1.0
FRUITS = 1.0
VEGGIES = 1.0
HVY_ALCOHOL = 0.0

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")

    models = {
        'Diabetes'        :  tf.keras.models.load_model(os.path.join(model_dir, "Diabetes_binary_model.keras")),
        'Penyakit Jantung':  tf.keras.models.load_model(os.path.join(model_dir, "HeartDiseaseor_model.keras")),
        'Stroke'          :  tf.keras.models.load_model(os.path.join(model_dir, "Stroke_model.keras")),
        'Hipertensi'      :  tf.keras.models.load_model(os.path.join(model_dir, "HighBP_model.keras")),
        'Kolesterol Tinggi':  tf.keras.models.load_model(os.path.join(model_dir, "HighChol_model.keras"))
    }

    scalers = {
        'Diabetes'        :  joblib.load(os.path.join(model_dir, "scaler_Diabetes_binary.pkl")),
        'Penyakit Jantung':  joblib.load(os.path.join(model_dir, "scaler_HeartDiseaseorAttack.pkl")),
        'Stroke'          :  joblib.load(os.path.join(model_dir, "scaler_Stroke.pkl")),
        'Hipertensi'      :  joblib.load(os.path.join(model_dir, "scaler_HighBP.pkl")),
        'Kolesterol Tinggi': joblib.load(os.path.join(model_dir, "scaler_HighChol.pkl"))
    }

    thresholds = {
        'Diabetes'         : 0.42,
        'Penyakit Jantung' : 0.46,
        'Stroke'           : 0.45,
        'Hipertensi'       : 0.52,
        'Kolesterol Tinggi': 0.50
    }

    fitur = [
        'Age', 'Sex', 'BMI', 'GenHlth', 'MentHlth', 'PhysHlth',
        'DiffWalk', 'CholCheck', 'Smoker', 'PhysActivity',
        'Fruits', 'Veggies', 'HvyAlcoholConsump'
    ]

    raw_data_dict = {
        'Age': [AGE], 'Sex': [SEX], 'BMI': [BMI], 'GenHlth': [GEN_HLTH], 
        'MentHlth': [MENT_HLTH], 'PhysHlth': [PHYS_HLTH], 'DiffWalk': [DIFF_WALK], 
        'CholCheck': [CHOL_CHECK], 'Smoker': [SMOKER], 'PhysActivity': [PHYS_ACTIVITY], 
        'Fruits': [FRUITS], 'Veggies': [VEGGIES], 'HvyAlcoholConsump': [HVY_ALCOHOL]
    }
    
    df_input = pd.DataFrame(raw_data_dict)[fitur]

    print("\n=== HASIL DETEKSI RISIKO KESEHATAN ===")
    
    # Prediksi penyakit
    for nama_penyakit in models.keys():
        df_temp = df_input.copy()
        
        if nama_penyakit in ['Diabetes', 'Penyakit Jantung', 'Stroke']:
            # Kelompok diabetes, penyakit jantung dan stroke
            df_temp['BMI_Smoker'] = df_temp['BMI'] * df_temp['Smoker']
            df_temp['Unhealthy_Index'] = df_temp['BMI'] * (1 - df_temp['PhysActivity'])
            df_temp['Age_GenHlth'] = df_temp['Age'] * df_temp['GenHlth']
            fitur_lengkap = fitur + ['BMI_Smoker', 'Unhealthy_Index', 'Age_GenHlth']
            
        else: # Kelompok hipertensi dan kolestrol
            df_temp['Age_CholCheck'] = df_temp['Age'] * df_temp['CholCheck']
            df_temp['Stress_Physical_Index'] = df_temp['MentHlth'] * df_temp['PhysHlth']
            df_temp['BMI_Alcohol'] = df_temp['BMI'] * df_temp['HvyAlcoholConsump']
            fitur_lengkap = fitur + ['Age_CholCheck', 'Stress_Physical_Index', 'BMI_Alcohol']

        # Mengekstrak dataframe menjadi matriks numpy array 16 kolom
        input_data = df_temp[fitur_lengkap].values
        
        # Menerapkan standarisasi fitur menggunakan scaler penyakit tersebut
        input_scaled = scalers[nama_penyakit].transform(input_data)
        
        # Menjalankan prediksi model antara rentang 0.0 - 1.0
        input_tensor = tf.convert_to_tensor(input_scaled, dtype=tf.float32)
        prediction = models[nama_penyakit](input_tensor, training=False)
        probabilitas = float(prediction[0][0])
        
        # Membandingkan probabilitas dengan threshold kustom masing-masing penyakit
        status_risiko = "Tinggi" if probabilitas > thresholds[nama_penyakit] else "Rendah"
        
        # Output hasil prediksi 
        print(f"\nDisease: {nama_penyakit}")
        print(f"  Probability Score : {probabilitas:.4f}")
        print(f"  Optimal Threshold : {thresholds[nama_penyakit]:.2f}")
        print(f"  Predicted Risk    : {status_risiko}")

if __name__ == "__main__":
    main()