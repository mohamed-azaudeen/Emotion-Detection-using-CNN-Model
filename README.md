# Emotion-Detection from uploaded images

# EDM30LayerCNN - Emotion Detection Model

This project implements a **30-layer Convolutional Neural Network (CNN)** for **facial emotion recognition** using PyTorch.  
The model is trained on grayscale facial images and classifies them into **7 emotion categories**:  
`angry, disgust, fear, happy, neutral, sad, surprise`.

---

## 🚀 Features
- **Custom 30-Layer CNN (EDM30LayerCNN)**  
  Deep architecture with convolution, batch normalization, ReLU activation, and max pooling layers.
  
- **Preprocessing**  
  - Converts images to grayscale.  
  - Resizes to `224x224`.  
  - Normalizes with dataset-specific mean and standard deviation.  

- **Training**  
  - Optimizer: Adam (`lr=0.001`)  
  - Loss Function: CrossEntropyLoss  
  - Epochs: 100  
  - Batch Size: 64  

- **Evaluation**  
  - Prints training loss every 10 epochs.  
  - Computes test accuracy after training.  

- **Prediction**  
  - Loads trained model weights (`edm30_model.pth`).  
  - Predicts emotion class for a given input image.  
  - Displays the image with predicted label using Matplotlib.  

---

## 📂 Dataset
- **Train Path**: `data/train`  
- **Test Path**: `data/test`  
- Images are organized in folders by class (e.g., `happy/`, `sad/`, etc.).  
- Example input image: `sample images/sad.jpg`.

---

## 🧠 Model Architecture
- **Convolutional Layers**: 15 blocks with increasing channels (16 → 256).  
- **Batch Normalization + ReLU** after each convolution.  
- **MaxPooling** applied every 3 layers.  
- **Adaptive Average Pooling** reduces spatial dimensions to `(1,1)`.  
- **Fully Connected Layer** maps features to 7 emotion classes.  

---

## 📊 Training Results
- Example training loss progression:

- **Test Accuracy**: ~53.6% (on sample dataset).

---
