from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from io import BytesIO
import os
import gdown

app = Flask(__name__, static_folder='static')

MODEL_PATH = 'model/mango_leaf_disease_simple.h5'
MODEL_URL = 'https://drive.google.com/uc?id=b21tOpUAAAsWuq0ElDxPF_SPlSA'

# Pastikan model diunduh
if not os.path.exists(MODEL_PATH):
    os.makedirs('model', exist_ok=True)
    try:
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")
    except Exception as e:
        print("Failed to download model:", e)

# Load model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    model = None

# Kelas prediksi
classes = [
    'Anthracnose',
    'Bacterial Canker',
    'Cutting Weevil',
    'Die Back',
    'Gall Midge',
    'Healthy',
    'Powdery Mildew',
    'Sooty Mould'
]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_label = classes[predicted_index]
        confidence = float(prediction[predicted_index])

        return jsonify({
            'label': predicted_label,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve frontend
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
