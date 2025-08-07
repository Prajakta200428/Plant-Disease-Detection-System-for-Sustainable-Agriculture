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

## 📁 Dataset used for project
Access the dataset used in this project below:
- **Download link:**  [Kaggle: New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- To use this dataset, you'll need a Kaggle account. Once downloaded, extract the archive and upload the dataset on your the google drive.
- The dataset is divided into `train` and `valid` directories.  
- Each class folder contains multiple images of diseased and healthy plant leaves.  
- Total classes: **38**.  

---

## Requirements
- streamlit
- tensorflow
- numpy
- pillow
- matplotlib
- seaborn
- ngrok

---

## 🧠 Model Architecture

The CNN model consists of:

- 4 Convolutional layers with increasing filter sizes (32 → 256)  
- 3 MaxPooling layers to downsample  
- Flatten layer followed by 2 Dense layers  
- Dropout for regularization  
- Final Dense layer with `softmax` activation for multiclass classification  

---

## 🙋‍♀️ How It Works

1. **Upload a leaf image** (`.jpg` / `.png`/`.jpeg`)
2. The model processes the image and predicts the **disease class**
3. The app displays:
   - ✅ The **name of the disease**
   - 📝 A **solution** specific to the plant disease
4. If you want to print the predicted disease output, the **print option** is available

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

---

## 🧮 Metrics Tracked

- **Accuracy**  
- **Precision**  
- **Recall**  
- **Loss**

---

## 💡 Future Improvements

- 📷 Add **camera support** for mobile users  
- 🎯 Improve model accuracy using **transfer learning**  
- 🗣️ Add **voice-based disease diagnosis**  
- 🌍 **Localize** solution text in regional languages
