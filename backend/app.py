from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from io import BytesIO
from flask import send_from_directory

app = Flask(__name__)

model = load_model('model/mango_leaf_disease_simple.h5')

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
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')  # Pastikan RGB
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)  # ⬅️ Ini lebih aman daripada np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_label = classes[predicted_index]
        confidence = float(prediction[predicted_index])

        print("Prediction array:", prediction)
        print("Max confidence:", np.max(prediction))

        return jsonify({
            'label': predicted_label,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_ui():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
