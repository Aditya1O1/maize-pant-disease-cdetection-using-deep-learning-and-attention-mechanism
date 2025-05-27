import os
import logging
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from tensorflow.keras.models import load_model
from model.multihead_attention_block import MHSABlock  # Custom Layer

# Logging Setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load Disease Detection Model
MODEL_PATH = r"C:\Users\lenovo\Desktop\corn_leaf_disease_classifier\model\finetuned_best_model (1).keras"
disease_model = load_model(MODEL_PATH, compile=False, custom_objects={'MHSABlock': MHSABlock})
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Load Binary Classifier (Corn vs Non-Corn)
resnet_model = models.resnet18(pretrained=True)
for param in resnet_model.parameters():
    param.requires_grad = False
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
RESNET_MODEL_PATH = r"C:\Users\lenovo\Desktop\corn_leaf_disease_classifier\model\model.pth"
resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=torch.device('cpu')))
resnet_model.eval()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image_for_keras(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_corn_or_not(image_path):
    """
    Predicts whether the given image is of a corn leaf or not.
    Returns:
        predicted_class (int): 0 for Corn, 1 for Not Corn
        confidence (float): probability score for the predicted class
    """
    img = Image.open(image_path).convert('RGB')
    img = resnet_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(img)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        logging.debug(f"Binary Prediction: {predicted.item()} with confidence: {confidence.item()}")
    return predicted.item(), confidence.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            logging.warning("No file part in request.")
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            logging.warning("No file selected.")
            return render_template('index.html', error="No file selected")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logging.info(f"File saved to {filepath}")

            corn_check, binary_confidence = predict_corn_or_not(filepath)

            # ✅ Fixed logic (assuming 0 = Corn, 1 = Not Corn)
            binary_result = 'Corn' if corn_check == 0 else 'Not Corn'
            image_url = f'/static/uploads/{filename}'

            if binary_result == 'Not Corn':
                logging.info("Image is not corn.")
                return render_template('index.html',
                                       binary_result=binary_result,
                                       binary_confidence=round(binary_confidence * 100, 2),
                                       image_url=image_url)

            # Predict disease
            img = prepare_image_for_keras(filepath)
            preds = disease_model.predict(img)[0]

            top_index = np.argmax(preds)
            predicted_class = CLASS_NAMES[top_index]
            confidence_score = round(preds[top_index] * 100, 2)

            # Top 3 predictions excluding the top one
            top_indices = preds.argsort()[-3:][::-1]
            alternate_guesses = [(CLASS_NAMES[i], round(preds[i] * 100, 2))
                                 for i in top_indices if i != top_index]

            logging.info(f"Predicted: {predicted_class} ({confidence_score}%)")

            return render_template('index.html',
                                   binary_result=binary_result,
                                   binary_confidence=round(binary_confidence * 100, 2),
                                   predicted_class=predicted_class,
                                   confidence_score=confidence_score,
                                   alternate_guesses=alternate_guesses,
                                   image_url=image_url)

        else:
            logging.warning("Invalid file type.")
            return render_template('index.html', error="File type not allowed. Upload PNG, JPG, or JPEG.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
