import os
from flask import Flask, request, jsonify, render_template
import torch
import tensorflow as tf
from PIL import Image
import numpy as np
from torchvision import transforms
from models.cnn import CNN1
from models.tf_cnn import create_cnn2

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch model
pytorch_model = CNN1().to(device)
try:
    pytorch_model.load_state_dict(torch.load("MAGATTE_FALL_model.torch", map_location=device))
    pytorch_model.eval()
except Exception as e:
    print(f"Error loading PyTorch model: {e}")

# Load TensorFlow model
try:
    tf_model = tf.keras.models.load_model("MAGATTE_FALL_model.tensorflow")
except:
    try:
        tf_model = create_cnn2()
        tf_model.load_weights("MAGATTE_FALL_model.tensorflow")
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        tf_model = None

# ✅ Liste des classes CORRECTEMENT réordonnée
classes = ['glioma', 'meningioma', 'notumor', 'pituitari']

# Preprocessing for PyTorch
pytorch_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Preprocessing for TensorFlow
def preprocess_tf_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        model_name = request.form.get('model', 'CNN1').upper()
        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400

        image = Image.open(file.stream).convert('RGB')

        if model_name == 'CNN1':
            if not pytorch_model:
                return jsonify({'error': 'PyTorch model not loaded'}), 500

            img = pytorch_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = pytorch_model(img)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probabilities))
                confidence = float(probabilities[pred]) * 100

        elif model_name == 'CNN2':
            if not tf_model:
                return jsonify({'error': 'TensorFlow model not loaded'}), 500

            img = preprocess_tf_image(image)
            output = tf_model.predict(img)[0]
            probabilities = tf.nn.softmax(output).numpy()
            pred = int(np.argmax(probabilities))
            confidence = float(probabilities[pred]) * 100

        else:
            return jsonify({'error': 'Invalid model name'}), 400

        result = {
            'prediction': classes[pred],
            'confidence': f"{confidence:.2f}%",
            'all_probabilities': {classes[i]: f"{p * 100:.2f}%" for i, p in enumerate(probabilities)},
            'model_used': model_name,
            'status': 'success'
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
