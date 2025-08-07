# 🌿 Plant-Disease-Detection-System-for-Sustainable-Agriculture 

A deep learning-based web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN). Built with **TensorFlow**, **Streamlit**, and deployed using **Ngrok**, this project not only predicts the disease class but also provides detailed, actionable solutions for each plant disease.

---

## 📌 Features

- ✅ Upload a plant leaf image and detect the disease automatically  
- 🧠 Uses a custom-trained Convolutional Neural Network (CNN)  
- 🌱 Displays disease-specific remedies and agricultural advice  
- 📊 Real-time performance monitoring with metrics like accuracy, precision, and recall  
- 🚀 Easy deployment using Streamlit and Ngrok for instant web access  

---

## 📁 Dataset
- archive(2).zip
- The dataset is divided into `train` and `valid` directories.  
- Each class folder contains multiple images of diseased and healthy plant leaves.  
- Total classes: **38**.  

---

## 🧠 Model Architecture

The CNN model consists of:

- 4 Convolutional layers with increasing filter sizes (32 → 256)  
- 3 MaxPooling layers to downsample  
- Flatten layer followed by 2 Dense layers  
- Dropout for regularization  
- Final Dense layer with `softmax` activation for multiclass classification  

---

## 🛠️ Tech Stack

| Technology          | Description                        |
|---------------------|------------------------------------|
| **Python**          | Core programming language          |
| **TensorFlow/Keras**| Deep learning framework            |
| **Streamlit**       | Frontend UI for interaction        |
| **Ngrok**           | Public URL tunneling for Streamlit |
| **Matplotlib/Seaborn** | Visualization of training metrics |
| **OpenCV & PIL**    | Image processing libraries         |



streamlit
tensorflow
numpy
pillow
matplotlib
seaborn
ngrok

