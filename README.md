# ✍️ Handwritten Digit Recognition using CNN

## 📘 Overview  
This project is part of my **Machine Learning & Artificial Intelligence Internship** at **Cloudcredits Technologies**.  
The goal of this task is to build a **Convolutional Neural Network (CNN)** that can recognize handwritten digits (0–9) from images.  

The model is trained on a **Reduced MNIST dataset**, where each image represents a single handwritten digit.  
CNNs are highly effective for this task because they can automatically learn important visual patterns like edges, curves, and shapes from pixel data.

---

## 🎯 Objective  
To accurately classify handwritten digits into 10 categories (0–9) using a deep learning model based on convolutional layers.

---

## 🗂 Dataset  
- **Name:** Reduced MNIST Dataset  
- **Structure:**

- **Image Size:** 28×28 pixels  
- **Color Mode:** Grayscale (1 channel)  
- **Classes:** 10 (Digits 0 to 9)

---

## ⚙️ Project Workflow  

### 1️⃣ Data Preprocessing  
- Loaded images from directories using `ImageDataGenerator`.  
- Normalized pixel values from `[0, 255] → [0, 1]`.  
- Resized all images to `(28×28)` pixels.  
- Used one-hot encoding for labels (`class_mode='categorical'`).  

### 2️⃣ CNN Model Architecture  
| Layer | Description |
|-------|--------------|
| **Conv2D (32 filters)** | Extracts basic features like lines and curves |
| **MaxPooling2D** | Reduces image size while keeping dominant features |
| **Conv2D (64 filters)** | Learns more complex patterns |
| **MaxPooling2D** | Further reduces dimensionality |
| **Flatten** | Converts 2D matrices to 1D vectors |
| **Dense (128 neurons)** | Fully connected hidden layer |
| **Dense (10 neurons, Softmax)** | Output layer for 10 digit classes |

---

## 🧠 Model Training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** 20  
- **Metrics:** Accuracy  
- **Validation Data:** Reduced Testing Data  

📈 During training, both **training** and **validation accuracy** improved steadily, reaching over **98–99%**.

---

## 📊 Model Evaluation  

| Metric | Description | Score |
|---------|--------------|-------|
| **Accuracy** | Correctly predicted digits | ~0.99 |
| **Loss** | Difference between predicted and actual | ~0.03 |

✅ The model performs exceptionally well on unseen test images.

---

## 📉 Confusion Matrix  
A confusion matrix was plotted using `seaborn` to visualize model performance:


> Nearly all values are concentrated along the diagonal, showing high classification accuracy.

---

## 🖼️ Sample Predictions  
Below are sample images from the test set with their true and predicted labels:


All predictions were accurate for the shown samples.

---

## 💾 Model Saving  
The trained model was saved as:
```bash
mnist_cnn_model.h5

git clone https://github.com/<your-username>/Cloudcredits.git
cd "Cloudcredits/Hand Written Digit Classifier"
pip install -r requirements.txt
python HandWritten_digit_Classifier.py






   
