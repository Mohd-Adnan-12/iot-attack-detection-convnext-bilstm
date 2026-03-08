# 🛡️ IoT Network Attack Detection using ConvNeXt + Bidirectional LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-97.96%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-BoT--IoT%20(UNSW)-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A hybrid deep learning-based Intrusion Detection System (IDS) that protects IoT environments from network attacks using **ConvNeXt** for spatial feature extraction and **Bidirectional LSTM** for temporal sequence modeling.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Baseline Comparison](#baseline-comparison)
- [Future Work](#future-work)
- [References](#references)

---

## 🔍 Overview

IoT devices are resource-constrained and lack built-in security, making them attractive targets for attackers. Traditional signature-based IDS cannot detect zero-day or evolving threats.

This project implements an intelligent, automated network attack detection system that:
- Detects **5 attack types**: DDoS, DoS, Reconnaissance, Theft, and Normal traffic
- Trains on **3.6 million** real IoT network flow records
- Achieves **97.96% overall accuracy** on 733,705 test records
- Provides a **Streamlit web interface** for real-time batch predictions

---

## 🧠 Architecture

```
Input (25 features)
       │
   Reshape → (25, 1)
       │
ConvNeXt Block 1: Conv1D(64) → LayerNorm → GELU
       │
ConvNeXt Block 2: Conv1D(128) → LayerNorm → GELU
       │
Bidirectional LSTM (64 units × 2 directions = 128 total)
       │
  Dropout (0.3)
       │
  Dense (64, ReLU)
       │
  Dense (5, Softmax)
       │
Output: [DDoS | DoS | Normal | Recon | Theft]
```

### Why ConvNeXt + BiLSTM?
| Component | Role |
|-----------|------|
| **ConvNeXt Blocks** | Extract spatial correlations among 25 network flow features |
| **Bidirectional LSTM** | Capture sequential attack patterns across multiple flows (both directions) |
| **Hybrid Fusion** | Combines spatial + temporal representations for superior detection |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 0.0003) |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 10 (with Early Stopping, patience=3) |
| Batch Size | 512 |
| Platform | Google Colab (T4 GPU) |

---

## 📊 Dataset

**BoT-IoT** — Bot Internet of Things Dataset  
Source: University of New South Wales (UNSW), Canberra  
Kaggle: [`vigneshvenkateswaran/bot-iot`](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot)

| Property | Details |
|----------|---------|
| Total Records | ~72 million network flows |
| After 5% Sampling | ~3.6 million records |
| Training Set (80%) | ~2.9 million records |
| Test Set (20%) | 733,705 records |
| Features Used | 25 selected features |
| Attack Classes | 5 (DDoS, DoS, Normal, Reconnaissance, Theft) |

### Class Distribution
| Class | Count | % | Challenge |
|-------|-------|---|-----------|
| DDoS | ~1,926,655 | 52.5% | Dominant class |
| DoS | ~1,650,270 | 45.0% | Similar to DDoS |
| Reconnaissance | ~91,010 | 2.5% | Minority class |
| Normal | ~500 | 0.01% | Very rare |
| Theft | ~90 | 0.001% | Extremely imbalanced |

---

## 📈 Results

### Overall Performance
| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **97.96%** |
| Correct Predictions | 718,723 / 733,705 |
| Training Accuracy | ~98% |
| Validation Accuracy | ~98% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| DDoS | 0.98 | 0.98 | 0.98 | 98.2% |
| DoS | 0.97 | 0.98 | 0.97 | 97.6% |
| Normal | 0.86 | 0.85 | 0.85 | 85.0% |
| Reconnaissance | 0.99 | 1.00 | 0.99 | 99.7% ⭐ |
| Theft | 0.00 | 0.00 | 0.00 | 0.0% ⚠️ |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | **97.96%** |

> ⚠️ Theft detection failed due to extreme class imbalance (only 18 test samples). Future work includes SMOTE oversampling to address this.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/iot-attack-detection-convnext-bilstm.git
cd iot-attack-detection-convnext-bilstm

# Install dependencies
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)
```
tensorflow>=2.10
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
streamlit>=1.15
matplotlib>=3.5
seaborn>=0.11
```

---

## 🚀 Usage

### Training the Model
```python
# Run in Google Colab (recommended for GPU)
python train.py
```

### Running the Streamlit App
```bash
streamlit run app.py
```

The web interface supports:
- **CSV Batch Prediction** — Upload a CSV of network flows
- **Manual Input** — Enter feature values manually
- **Accuracy Comparison** — View model vs baseline comparison
- **Download Results** — Export predictions as CSV

### Making Predictions
```python
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("convnext_bilstm_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocess your input (25 features)
X = scaler.transform(your_features)
X = X.reshape(-1, 25, 1)

pred = model.predict(X)
label = label_encoder.inverse_transform([np.argmax(pred)])
print(f"Predicted Attack: {label[0]}")
```

---

## 📁 Project Structure

```
iot-attack-detection-convnext-bilstm/
│
├── train.py                  # Model training script
├── app.py                    # Streamlit web interface
├── preprocess.py             # Data loading & preprocessing pipeline
├── model.py                  # ConvNeXt + BiLSTM architecture definition
├── evaluate.py               # Evaluation & metrics generation
│
├── models/
│   ├── convnext_bilstm_model.h5   # Saved trained model
│   ├── scaler.pkl                 # StandardScaler
│   └── label_encoder.pkl          # LabelEncoder
│
├── notebooks/
│   └── ConvNeXt_BiLSTM_IoT.ipynb  # Google Colab training notebook
│
├── requirements.txt
└── README.md
```

---

## 📊 Baseline Comparison

| Model | Type | Accuracy |
|-------|------|----------|
| Naive Bayes | Traditional ML | 71.2% |
| Decision Tree | Traditional ML | 89.7% |
| Random Forest | Ensemble ML | 92.3% |
| CNN only | Deep Learning | 95.1% |
| LSTM only | Deep Learning | 94.8% |
| Standard CNN + LSTM | Hybrid DL | 96.5% |
| **ConvNeXt + BiLSTM (Ours)** | **Hybrid DL** | **97.96% ✅** |

Our model outperforms Random Forest by **+5.66%** and standard CNN+LSTM by **+1.46%**.

---

## 🔮 Future Work

- **SMOTE Oversampling** — Fix Theft detection (currently 0%) using synthetic data generation
- **Real-time Detection** — Integrate CICFlowMeter/Zeek for live network flow extraction
- **Edge Deployment** — Quantize and deploy on Raspberry Pi for on-device IoT protection
- **Federated Learning** — Train across distributed IoT devices without sharing raw data
- **Transformer Models** — Explore BERT-based architectures for network traffic
- **Explainability** — Add SHAP values to identify which features trigger detection

---

## 📚 References

1. Koroniotis et al. (2019). *Towards the Development of Realistic Botnet Dataset in the IoT for Network Forensic Analytics: Bot-IoT Dataset.* Future Generation Computer Systems.
2. Liu et al. (2022). *A ConvNet for the 2020s (ConvNeXt).* IEEE/CVF CVPR.
3. Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.

---

## 📄 License

This project is licensed under the MIT License.

---

*Mini Project 3 — Deep Learning for Cybersecurity*
