# 🧠 Tuberculosis & Pneumonia Detection from Chest X-Rays

A complete deep learning pipeline to detect **Tuberculosis (TB)** and **Pneumonia** from chest X-ray images using **transfer learning**, **Grad-CAM explainability**, and an interactive **Streamlit web application**.

---

## 📌 Project Highlights

- ✅ Uses **real-world chest X-ray data** (DICOM & JPEG)
- 🧠 **Transfer Learning** with ResNet18, ResNet34, DenseNet121
- 🧪 **Model Benchmarking** based on AUC & Accuracy
- 📊 **Performance metrics**: Accuracy, F1, ROC-AUC, Confusion Matrix
- 🖼️ **Grad-CAM visualizations** for model transparency
- 🌐 **Streamlit Web App** for live predictions

---

## 📁 Dataset Overview

- **Source**: National Institute of Tuberculosis and Respiratory Diseases (NITRD), New Delhi
- **Data Format**: JPEG (DA) & DICOM (DB)
- **Total Images**: 278
  - 125 TB Positive
  - 153 TB Negative
- **Labeling Rule**: Filenames starting with:
  - `p` → TB Positive
  - `n` → TB Negative

---

## 🧱 Directory Structure
```
project_root/
│
├── images/
│   ├── da/       # JPEG X-rays
│   └── db/       # DICOM X-rays
│
├── app.py                        # Streamlit frontend
├── chest_xray_prediction.ipynb  # Notebook: training & evaluation
├── saved_model.pt               # Trained PyTorch model
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ How to Use

### 🔹 1. Clone the Repo
```bash
git clone https://github.com/your-username/tb-xray-detection.git
cd tb-xray-detection
```

### 🔹 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 🔹 3. Train the Model (Optional)
Open the notebook and run all cells:
```bash
jupyter notebook chest_xray_prediction.ipynb
```
This will train and save `saved_model.pt`.

### 🔹 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Upload an image and view prediction + Grad-CAM heatmap.

---

## 📊 Sample Output

- ✅ **Prediction**: TB Positive (Probability: 0.91)
- 📍 **Explainability**: Grad-CAM shows affected lung region

---

## 🧪 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC Curve & AUC
- Confusion Matrix

---

## 📦 Tech Stack
- PyTorch
- Torchvision
- NumPy, Matplotlib, Seaborn
- scikit-learn
- Streamlit
- PIL, pydicom

---

## 🚀 Future Work
- Pneumonia multi-class support
- Model compression for mobile use
- Clinical validation with larger datasets

---

> 📁 This project was developed for academic purposes and showcases how adaptive deep learning models can be applied to real-world medical image data using responsible AI and interpretability techniques.
