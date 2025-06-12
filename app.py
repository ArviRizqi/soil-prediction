from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load model dan encoder
model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load(open('model/label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', inputs={}, prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        inputs = {key: request.form[key] for key in request.form}

        # Urutan fitur harus sesuai dengan training
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features = [float(inputs[key]) for key in feature_order]

        # Skala input
        features_scaled = scaler.transform([features])  # Hasil: (1, 7)

        # Prediksi kelas (hasil berupa array, contoh: [3])
        prediction = model.predict(features_scaled)     # Masih dalam bentuk label numerik

        # Konversi kembali ke label asli (misal: 'apple', 'banana', dll)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Tampilkan hasil di HTML
        return render_template('index.html', prediction_text=f"Rekomendasi tanaman: {predicted_label}", inputs=inputs)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", inputs=request.form)

if __name__ == '__main__':
    app.run(debug=True)

