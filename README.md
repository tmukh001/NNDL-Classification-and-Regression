# 🧠 Neural Networks & Deep Learning: Audio Classification & Housing Price Prediction

This project showcases my ability to build and evaluate **deep learning models for real-world tabular and audio data**, using both standard and advanced architectures. Completed as part of the *SC4001: Neural Networks and Deep Learning* course at NTU Singapore, this assignment is divided into two parts:

- 🎵 **Part A**: Music genre classification from engineered audio features  
- 🏘️ **Part B**: Housing price regression using multi-source property data in Singapore

---

## 🎵 Part A: Audio Genre Classification (Blues vs Metal)

### 🔍 Problem

Given engineered features from preprocessed audio snippets (from the [GTZAN dataset](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)), the task is to classify whether a track belongs to the **blues** or **metal** genre.

### 🧠 Techniques Used

- Feedforward DNN with 3 hidden layers (128 ReLU units each)
- Dropout regularization (p=0.2)
- Early stopping on validation loss
- Feature scaling using `StandardScaler`
- 5-fold cross-validation for:
  - Optimal batch size selection
  - Hidden layer size tuning
- Model explanation with **SHAP** (local feature importance)

### 📈 Results

- Achieved >90% accuracy with optimized batch size and neuron count
- Identified key features influencing genre prediction (e.g., MFCCs, chroma)
- SHAP force plots used to explain prediction on a test audio sample

---

## 🏘️ Part B: HDB Resale Price Prediction (Singapore)

### 🔍 Problem

Predict resale prices of public housing using real estate and transport-based features. Data sourced from [data.gov.sg](https://data.gov.sg).

### 📦 Features Used

- **Numerical**: Distance to MRT stations, lease remaining, floor area, centrality scores  
- **Categorical**: Town, month, storey range, flat type

### 📊 B1: PyTorch Tabular Model

- One hidden layer (50 neurons)
- Auto-tuned learning rate, Adam optimizer
- Trained on pre-2021 data, tested on 2021 data
- Reported metrics: RMSE and R²  
- Analysis of top 25 largest prediction errors

### ⚙️ B2: PyTorch Wide & Deep

- Two hidden layers (200 → 100 neurons)
- Used `TabPreprocessor` and `TabMlp`
- Batch size = 64, trained for 60 epochs
- Better performance vs B1 on test data

### 🧠 B3: Model Explainability with Captum

- Used only numeric features
- Applied multiple XAI techniques:
  - Integrated Gradients
  - SHAP
  - DeepLift
- Identified top 3 impactful features
- Interpreted directionality of feature effects

### 🚨 B4: Model Drift Analysis

- Tested model on 2022 & 2023 data to evaluate performance degradation
- Applied `Alibi Detect` to detect drifted features
- Discussed **covariate and concept drift**
- Suggested retraining as a mitigation strategy and demonstrated R² improvement

---

## 📁 Files & Structure

- `audio_gtzan.csv` — Audio features dataset  
- `hdb_price_prediction.csv` — Property data  
- `common_utils.py` — Preprocessing and model utility functions  
- `Classification.ipynb` — Part A notebook (audio genre classification)  
- `Regression.ipynb` — Part B notebook (housing price regression + drift)  

---

## 🛠️ Libraries & Tools

- Python, PyTorch, NumPy, Pandas  
- Librosa (audio feature extraction)  
- SHAP, Captum (model explanation)  
- PyTorch Tabular & PyTorch-WideDeep  
- Alibi Detect (drift detection)

---

## 👨‍💻 Author

**Tathagato Mukherjee**  
BSc (Hons), Data Science & AI, NTU Singapore

---
