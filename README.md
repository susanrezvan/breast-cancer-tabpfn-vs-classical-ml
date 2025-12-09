# Breast Cancer Classification  
## Classical Machine Learning Models vs. TabPFN Foundation Model

This project presents a comparative study between classical machine learning models —  
**K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Random Forest** —  
and the modern **TabPFN (Tabular Prior-Data Fitted Network)** foundation model for breast cancer diagnosis.

Using the **Breast Cancer Wisconsin Diagnostic Dataset (WBCD)**, we evaluate all models based on  
**Accuracy**, **Precision**, **Recall**, and **F1-Score**, both **with and without Feature Selection (Mutual Information)**.

---

## 📌 Key Findings

### 🔹 KNN
- Strongest classical model overall  
- With cosine distance + feature selection → **no false negatives**  
- Excellent recall, ideal for medical diagnosis

### 🔹 SVM
- **RBF kernel** delivers best performance without feature selection  
- After feature selection → **linear kernel** performs best  
- Polynomial kernel shows underfitting

### 🔹 Random Forest
- Extremely high precision (rare false positives)  
- Slightly decreased performance after feature selection

### 🔹 TabPFN (Best Model)
- Achieved the **highest accuracy (~98.25%)**  
- No need for hyperparameter tuning  
- Fast inference and stable performance  
- Outperformed all classical models

---

## 📂 Project Structure

breast-cancer-tabpfn-vs-classical-ml/
│
├── notebooks/                # All Jupyter notebooks for experiments
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── preprocessing.ipynb   # Cleaning, scaling, feature selection
│   ├── KNN.ipynb             # K-Nearest Neighbors experiments
│   ├── SVM.ipynb             # Support Vector Machine experiments
│   ├── RandomForest.ipynb    # Random Forest experiments
│   └── TabPFN.ipynb          # TabPFN model evaluation
│
├── data/                     # Dataset (not included in the repo)
│
├── results/                  # Exported plots, confusion matrices (optional)
│
├── models/                   # Saved ML models (optional)
│
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies (optional)



---

## 🧠 Methods

### 1️⃣ Preprocessing
- Handling missing values  
- Label encoding  
- Z-score standardization  
- Optional feature selection using **Mutual Information**

### 2️⃣ Classical ML Models
- **KNN** (k = 3, 5, 7, 9, 11) with Euclidean & Cosine metrics  
- **SVM** with Linear, Polynomial, RBF, Sigmoid kernels  
- **Random Forest** with default hyperparameters

### 3️⃣ TabPFN Model
- Prior-trained transformer-like model  
- No training required  
- Runs using a single inference pass  
- Input: (X_train, y_train, X_test)

---

## 📊 Evaluation Metrics

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**

Recall is especially important for medical diagnosis (minimizing false negatives).

---

## ⭐ Summary of Results

| Model | Best Accuracy | Notes |
|-------|--------------|-------|
| **KNN** | ~97.3% | Best classical baseline, high recall |
| **SVM (RBF)** | ~98% | Best classical model without feature selection |
| **Random Forest** | ~97.4% | Very high precision |
| **TabPFN** | **98.25%** | Best overall performance |

---

## 📘 Dataset

- **Breast Cancer Wisconsin Diagnostic Dataset (WBCD)**  
- 569 samples, 30 numerical features  
- Classes: **Benign (B)** and **Malignant (M)**  
- Features include: radius_mean, texture_mean, perimeter_mean, area_mean, etc.

---

## 🚀 How to Run

Install Dependencies

You will need:

- Python 3.9+  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  
- tabpfn  

Install using pip:

pip install pandas numpy scikit-learn seaborn matplotlib tabpfn
----
Open the Notebook:
Google colab
note:All notebooks were originally created in Google Colab, but they follow the standard Jupyter Notebook format.
---
Clone the Repository: 
git clone https://github.com/susanrezvan/breast-cancer-tabpfn-vs-classical-ml.git

