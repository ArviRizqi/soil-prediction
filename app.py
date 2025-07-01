import streamlit as st
import pickle
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load(open('model/label_encoder.pkl', 'rb'))

st.title('Prediksi Rekomendasi Tanaman')

# Form input
st.write('Masukkan nilai parameter berikut:')

feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
inputs = {}

for feature in feature_order:
    inputs[feature] = st.text_input(f"{feature}", "")

# Tombol prediksi
if st.button('Prediksi'):
    try:
        # Konversi input ke float sesuai urutan fitur
        features = [float(inputs[key]) for key in feature_order]

        # Skala input
        features_scaled = scaler.transform([features])  # (1,7)

        # Prediksi
        prediction = model.predict(features_scaled)

        # Konversi label numerik ke label asli
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Rekomendasi tanaman: {predicted_label}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
