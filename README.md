# 🌽 Maize Leaf Disease Detection Using Deep Learning and Attention Mechanism

1. Maize Leaf Disease Detection using Deep Learning and Attention Mechanism
Designed and deployed a two-stage deep learning web application for automated maize leaf disease detection, supporting large-scale, healthy maize cultivation in Bihar, crucial for food security and ethanol-based biofuel production. 
   Stage 1: Achieved 98.87% test accuracy using a pre-trained ResNet-18 CNN for binary classification (Maize vs. Non-Maize), effectively filtering 
            out irrelevant inputs. 
   Stage 2: Built a custom CNN with Multi-Head Self-Attention (MHSA) for multi-class disease classification (Blight, Common Rust, Gray Leaf Spot, 
            Healthy), achieving 91.57% test accuracy with strong F1-scores (Weighted F1-score of 0.93 
            (validation) and 0.92 (test) across four maize leaf classes).




## 🧠 Project Overview

This project aims to automate maize leaf disease identification using a hybrid deep learning approach combining **Convolutional Neural Networks (CNNs)** with **Multi-Head Self-Attention (MHSA)**. It is divided into two stages:

### 📌 Stage 1 – Binary Classification (Corn vs Non-Corn)
- Model: **Pre-trained ResNet-18** (PyTorch)
- Purpose: Filters out irrelevant or non-corn images before disease classification.
- Accuracy: **98.87% test accuracy**

### 📌 Stage 2 – Multi-class Disease Classification
- Model: **Custom CNN + MHSA** (Keras/TensorFlow)
- Classes:
  - Blight  
  - Common Rust  
  - Gray Leaf Spot  
  - Healthy
- Performance:  
  - **Test Accuracy**: 91.57%  
  - **F1 Scores**: Weighted F1 = 0.93 (Validation), 0.92 (Test)

---

## 🚀 Key Features

- 🔍 **Automated Leaf Detection**: Upload, validate, classify in seconds.
- 📊 **Two-Stage Classifier**: Filters irrelevant inputs for better reliability.
- 🎯 **Attention Mechanism**: Improves spatial-context learning.
- 🖼️ **Image Preview & Confidence Scores**: Displays both predictions and confidence levels.
- 📁 **User-Friendly Interface**: Built with Flask + HTML templates.

---

## 🧰 Technologies & Tools Used

- **Python**
- **Flask** (Web Application)
- **PyTorch** (ResNet-18 Binary Classifier)
- **TensorFlow/Keras** (Custom CNN + MHSA)
- **NumPy, PIL, TorchVision** (Preprocessing)
- **HTML/CSS** (Frontend)
- **Google Drive** (Model file delivery)

---

## 📁 Project Structure
corn-leaf-disease-app/
├── static/
│ └── images/
│ └── corn_image.avif # Sample image (optional)
├── templates/
│ ├── index.html # Main upload + result page
│ └── result.html # Optional result template
├── models/
│ ├── model.pth # ResNet-18 binary classifier (PyTorch)
│ ├── finetuned_best_model (1).keras # Disease classifier 
│ └── multihead_attention_block.py # Custom MHSA implementation
├── app.py # Flask application entry point
├── requirements.txt # All Python dependencies
├── .gitattributes # Git LFS configuration
├── README.md # Project description and usage
└── render.yaml # Optional Render deployment settings




## 🛠️ Installation & Setup

> ⚠️ Due to GitHub’s 25MB upload limit, model files are provided separately via Google Drive.
> Link: Very Soon I will upload complete driive link. Thanks for your patience.

### 📦 Prerequisites
Make sure Python ≥ 3.8 is installed. Then install required packages:




📁 Download Models
Download the model files from Google Drive:
xtract them and place inside the models/ directory as follows:
models/
├── model.pth
├── finetuned_best_model (1).keras
└── multihead_attention_block.py


▶️ Run the App
python app.py

Then open your browser and visit:
📍 http://127.0.0.1:5000


📸 Sample Usage
Upload an image of a maize (corn) leaf.

The app first checks if it's a corn image.

If valid, it classifies the disease among 4 classes.

Results are shown with confidence levels.



💡 Future Enhancements
Deployment to Hugging Face / Streamlit / Render

Mobile app integration

Dataset expansion and model fine-tuning


🤝 Credits
Author: Aditya Kumar Pandey

Institute: IIIT Bhagalpur

Domain: Deep Learning, Computer Vision, Agriculture Tech
