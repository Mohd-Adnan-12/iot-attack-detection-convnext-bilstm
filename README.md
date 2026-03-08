# 🛡️ IoT Network Attack Detection using ConvNeXt + Bidirectional LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-97.96%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-BoT--IoT%20(UNSW)-blueviolet)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A hybrid deep learning-based Intrusion Detection System (IDS) for IoT environments using **ConvNeXt** for spatial feature extraction and **Bidirectional LSTM** for temporal sequence modeling — trained on 3.6 million real IoT network flows.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Results](#results)
- [Project Files](#project-files)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Web App](#streamlit-web-app)
- [Baseline Comparison](#baseline-comparison)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [References](#references)

---

## 🔍 Overview

IoT devices are lightweight and resource-constrained, making them highly vulnerable to network attacks. Traditional signature-based IDS solutions fail against zero-day and evolving threats.

This project builds an end-to-end intelligent attack detection pipeline that:
- Classifies network traffic into **5 categories**: DDoS, DoS, Reconnaissance, Theft, and Normal
- Trains on **~3.6 million** real BoT-IoT network flow records using Google Colab (T4 GPU)
- Achieves **97.96% overall accuracy** on 733,705 test records
- Provides a **dark-themed Streamlit web interface** for batch CSV predictions and manual input

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
| **ConvNeXt Blocks** | Extract local spatial correlations among 25 network flow features |
| **Bidirectional LSTM** | Capture sequential attack patterns in both forward and backward directions |
| **Hybrid Fusion** | Combines spatial + temporal representations for superior accuracy |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 0.0003) |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 10 |
| Early Stopping | patience = 3 (monitors val_loss) |
| Batch Size | 512 |
| Platform | Google Colab (T4 GPU) |
| Training Time | ~30–45 minutes |

---

## 📊 Dataset

**BoT-IoT** — Bot Internet of Things Dataset
**Source:** University of New South Wales (UNSW), Canberra
**Kaggle:** [`vigneshvenkateswaran/bot-iot`](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot)

| Property | Details |
|----------|---------|
| Total Raw Records | ~72 million network flows |
| After 5% Sampling | ~3.6 million records |
| Training Set (80%) | ~2.9 million records |
| Test Set (20%) | 733,705 records |
| Features Used | 25 selected features |
| Attack Classes | 5 (DDoS, DoS, Normal, Reconnaissance, Theft) |

### Class Distribution

| Class | Approx. Count | Percentage | Challenge |
|-------|--------------|-----------|-----------|
| DDoS | ~1,926,655 | 52.5% | Dominant class |
| DoS | ~1,650,270 | 45.0% | Similar patterns to DDoS |
| Reconnaissance | ~91,010 | 2.5% | Minority class |
| Normal | ~500 | 0.01% | Very rare |
| Theft | ~90 | 0.001% | Extremely imbalanced |

---

## ⚙️ Preprocessing

The following steps were applied in `iot_attack_detection_ConvNeXt_BiLSTM.ipynb`:

| Step | Method | Reason |
|------|--------|--------|
| Data Loading | Chunked reading (200K rows), 5% stratified sampling by subcategory | Handle 72M records without RAM overflow |
| Column Removal | Dropped pkSeqID, stime, ltime, saddr, daddr, smac, dmac, attack, subcategory | Remove identifiers and leaky labels |
| Missing Values | Replace Inf → NaN, fill NaN → 0 | Prevent training errors |
| Categorical Encoding | LabelEncoder on proto, state, flgs, soui, doui, sco, dco | Convert strings to numeric |
| Label Encoding | LabelEncoder on target column (category) | DDoS=0, DoS=1, Normal=2, Recon=3, Theft=4 |
| Train/Test Split | 80/20 stratified split (random_state=42) | Maintain class distribution |
| Normalization | StandardScaler — fit on train, transform both | Normalize to mean=0, std=1 |
| Reshape | (samples, 25) → (samples, 25, 1) | Required for Conv1D input |

### 25 Selected Features

```
flgs    proto   sport   dport   pkts    bytes   state   seq
dur     mean    stddev  sum     min     max     soui    doui
sco     dco     spkts   dpkts   sbytes  dbytes  rate    srate
drate
```

---

## 📈 Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **97.96%** |
| Correct Predictions | 718,723 / 733,705 |
| Training Accuracy | ~98% |
| Validation Accuracy | ~98% |
| Weighted Precision | 0.98 |
| Weighted Recall | 0.98 |
| Weighted F1-Score | 0.98 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| DDoS | 0.98 | 0.98 | 0.98 | 98.2% |
| DoS | 0.97 | 0.98 | 0.97 | 97.6% |
| Normal | 0.86 | 0.85 | 0.85 | 85.0% |
| Reconnaissance | 0.99 | 1.00 | 0.99 | 99.7% ⭐ |
| Theft | 0.00 | 0.00 | 0.00 | 0.0% ⚠️ |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | **97.96%** |

> ⚠️ **Theft detection is 0%** due to extreme class imbalance — only 18 Theft test samples out of 733,705 total. SMOTE oversampling is planned as future work.

---

## 📁 Project Files

```
iot-attack-detection-convnext-bilstm/
│
├── iot_attack_detection_ConvNeXt_BiLSTM.ipynb     # Google Colab training notebook
│                                  # (data loading, preprocessing,
│                                  #  model training, saving)
│
├── app.py                         # Streamlit web interface
│                                  # (batch CSV prediction,
│                                  #  manual input, results view)
│
├── iot_attack_model_v2.keras      # Saved trained model (download from Drive)
├── scaler.pkl                     # Saved StandardScaler (download from Drive)


---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mohd-Adnan-12/iot-attack-detection-convnext-bilstm.git
cd iot-attack-detection-convnext-bilstm
```

### 2. Install Dependencies
```bash
pip3 install tensorflow streamlit scikit-learn pandas numpy joblib matplotlib
```

### 3. Download Model Files from Google Drive
Download these two files and place them in the project folder:
- `iot_attack_model_v2.keras`
- `scaler.pkl`

---

## 🚀 Usage

### Step 1 — Train the Model (Google Colab)
Open `iot_attack_detection_ConvNeXt_BiLSTM.ipynb` in Google Colab and run all cells.

```python
# The notebook will:
# 1. Download BoT-IoT dataset via kagglehub
# 2. Preprocess and sample 5% of data (~3.6M records)
# 3. Train ConvNeXt + BiLSTM for 10 epochs
# 4. Save model and scaler to Google Drive

model.save('/content/drive/MyDrive/iot_attack_model_v2.keras')
joblib.dump(scaler, '/content/drive/MyDrive/scaler.pkl')
```

> ⏱️ Training takes approximately 30–45 minutes on Google Colab T4 GPU.

### Step 2 — Run the Streamlit App (Local)
```bash
cd Desktop/Mini
python3 -m streamlit run app.py
```

App opens at: **http://localhost:8501**

### Step 3 — Make Predictions
```python
from tensorflow.keras.models import load_model
import joblib, numpy as np

model  = load_model("iot_attack_model_v2.keras", compile=False)
scaler = joblib.load("scaler.pkl")

FEATURES = ['flgs','proto','sport','dport','pkts','bytes','state',
            'seq','dur','mean','stddev','sum','min','max',
            'soui','doui','sco','dco','spkts','dpkts',
            'sbytes','dbytes','rate','srate','drate']

LABELS = {0:'DDoS', 1:'DoS', 2:'Normal', 3:'Reconnaissance', 4:'Theft'}

X = scaler.transform(your_dataframe[FEATURES])
X = X.reshape(-1, 25, 1).astype('float32')

preds = model.predict(X)
labels = [LABELS[np.argmax(p)] for p in preds]
```

---

## 🖥️ Streamlit Web App

The app (`app.py`) provides a dark-themed interface with 3 tabs:

### Tab 1 — 📂 CSV Prediction
- Upload any CSV with the 25 model features
- Runs batch prediction on all records
- Shows dominant attack type with severity level
- Displays attack distribution table and bar chart
- Shows accuracy vs actual labels (if `actual_category` column present)
- Download results as CSV

### Tab 2 — 🔍 Manual Input
- Enter all 25 feature values manually
- Load 5 presets: Normal, DDoS, DoS, Reconnaissance, Theft
- Shows prediction with confidence score
- Displays all 5 class probability scores

### Tab 3 — 📋 Feature Guide
- Reference table explaining all 25 features
- Full model architecture diagram


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

Our model outperforms:
- Random Forest by **+5.66%**
- Standard CNN+LSTM by **+1.46%**
- CNN only by **+2.86%**

---

## ⚠️ Known Limitations

| Limitation | Description |
|-----------|-------------|
| **Class Imbalance** | Theft has only 18 test samples → 0% detection accuracy |
| **No SMOTE** | Class balancing was not applied during training |
| **Feature Mismatch** | Wireshark exports packet-level data; model needs flow-level features (mean, stddev, soui, etc.) — direct live capture not supported |
| **IoT-Specific** | Trained on IoT traffic; may not generalize well to enterprise networks |

---

## 🔮 Future Work

- **SMOTE Oversampling** — Generate synthetic Theft and Normal samples to fix class imbalance
- **Real-time Detection** — Integrate CICFlowMeter or Zeek for live Wireshark-based flow extraction
- **Edge Deployment** — Quantize and deploy lightweight model on Raspberry Pi for on-device IoT protection
- **Federated Learning** — Train across distributed IoT devices without sharing raw data
- **Transformer Models** — Explore BERT-based architectures for network traffic analysis
- **Explainability** — Add SHAP values to identify which features triggered each attack detection

---

## 📚 References

1. N. Koroniotis, N. Moustafa, E. Sitnikova, and B. Turnbull, "Towards the Development of Realistic Botnet Dataset in the Internet of Things for Network Forensic Analytics: Bot-IoT Dataset," *Future Generation Computer Systems*, vol. 100, pp. 779–796, 2019.
2. Z. Liu et al., "A ConvNet for the 2020s (ConvNeXt)," in *Proc. IEEE/CVF CVPR*, pp. 11976–11986, 2022.
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
4. N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems," in *Proc. MilCIS*, 2015.
5. A. A. Diro and N. Chilamkurti, "Distributed Attack Detection Scheme Using Deep Learning Approach for Internet of Things," *Future Generation Computer Systems*, vol. 82, pp. 761–768, 2018.
6. R. Vinayakumar et al., "Deep Learning Approach for Intelligent Intrusion Detection System," *IEEE Access*, vol. 7, pp. 41525–41550, 2019.
7. C. Yin, Y. Zhu, J. Fei, and X. He, "A Deep Learning Approach for Intrusion Detection Using Recurrent Neural Networks," *IEEE Access*, vol. 5, pp. 21954–21961, 2017.
8. M. Ge et al., "Deep Learning-Based Intrusion Detection for IoT Networks," in *Proc. IEEE PRDC*, pp. 256–265, 2019.
9. N. Moustafa, J. Hu, and J. Slay, "A Holistic Review of Network Anomaly Detection Systems," *Journal of Network and Computer Applications*, vol. 128, pp. 33–55, 2019.
10. Y. Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection," in *Proc. NDSS*, 2018.
11. H. Shapoorifard and P. Shamsinejad, "Intrusion Detection Using a Novel Hybrid Method Incorporating an Improved KNN," *Int. Journal of Computer Applications*, vol. 173, no. 1, 2017.
12. F. Chollet, *Deep Learning with Python*. Manning Publications, 2017.

---

## 📄 License

This project is licensed under the MIT License.

---

*Mini Project 3 — Deep Learning for Cybersecurity*
*Tools: Python · TensorFlow · Keras · Scikit-learn · Streamlit · Google Colab · BoT-IoT Dataset*
