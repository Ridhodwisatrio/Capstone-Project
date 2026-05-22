import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
# from sklearn.utils import class_weight as sklearn_class_weight

# Load Data
X_train = pd.read_csv('../data/X_train_smote_HeartDiseaseorAttack.csv')
y_train = pd.read_csv('../data/y_train_smote_HeartDiseaseorAttack.csv').values.flatten()
df_test = pd.read_csv('../data/data_test_clean.csv')
target = 'HeartDiseaseorAttack'
df_test_clean = df_test.dropna(subset=[target]).copy()

fitur = [
    'Age', 'Sex', 'BMI', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'CholCheck', 'Smoker', 'PhysActivity',
    'Fruits', 'Veggies', 'HvyAlcoholConsump'
]

X_train_df = X_train[fitur].copy()

# Feature Engineering
# Feature engineering untuk data train
X_train_df['BMI_Smoker'] = X_train_df['BMI'] * X_train_df['Smoker']
X_train_df['Unhealthy_Index'] = X_train_df['BMI'] * (1 - X_train_df['PhysActivity'])
X_train_df['Age_GenHlth'] = X_train_df['Age'] * X_train_df['GenHlth']

# Menerapkan feature engineering yang sama untuk data test
df_test_clean['BMI_Smoker'] = df_test_clean['BMI'] * df_test_clean['Smoker']
df_test_clean['Unhealthy_Index'] = df_test_clean['BMI'] * (1 - df_test_clean['PhysActivity'])
df_test_clean['Age_GenHlth'] = df_test_clean['Age'] * df_test_clean['GenHlth']

# Merubah menjadi matriks 16 kolom
fitur_lengkap = fitur + ['BMI_Smoker', 'Unhealthy_Index', 'Age_GenHlth']
X_train_raw = X_train_df[fitur_lengkap].values
X_test_raw = df_test_clean[fitur_lengkap].values
y_test = df_test_clean[target].values.astype(np.int32).flatten()

# Normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
joblib.dump(scaler, "../model/scaler_HeartDiseaseorAttack.pkl")

# Build Model
def build_model(input_dim):
    inputs = keras.layers.Input(shape=(input_dim,))
    dense1 = keras.layers.Dense(256, activation='relu')(inputs)
    dense1 = keras.layers.BatchNormalization()(dense1)
    dense1 = keras.layers.Dropout(0.4)(dense1)

    dense2 = keras.layers.Dense(128, activation='relu')(dense1)
    dense2 = keras.layers.BatchNormalization()(dense2)
    dense2 = keras.layers.Dropout(0.3)(dense2)

    dense3 = keras.layers.Dense(64, activation='relu')(dense2)
    outputs = keras.layers.Dense(1, activation='sigmoid')(dense3)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
    )
    return model

# Penerapan Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-5
)

callbacks_list = [early_stop, reduce_lr]

# Training dan Evaluasi Model
def train_model(X_train, y_train, X_test, y_test, epochs=50):

    model = build_model(input_dim=X_train.shape[1])

    print("\n--- Training Serangan Jantung---")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )
    print("--- Training Selesai ---\n")

    results = model.evaluate(X_test, y_test, verbose=1)

    test_loss = results[0]
    test_accuracy = results[1]

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.46).astype(int).flatten()

    target_names = ['Non-HeartDiseaseorAttack (Sehat)', 'HeartDiseaseorAttack']
    print("\n=== PERFORMANCE REPORT PENYAKIT SERANGAN JANTUNG ===")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Loss: {test_loss:.4f}")

    return model, history

trained_model, history = train_model(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, epochs=20)

# save model
trained_model.save("../model/HeartDiseaseor_model.keras")