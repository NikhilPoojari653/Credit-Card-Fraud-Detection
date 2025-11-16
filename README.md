# Credit Card Fraud Detection using Machine Learning

This project builds a robust machine learning system to detect fraudulent credit card transactions in a highly imbalanced dataset. It focuses on accurate fraud identification through effective preprocessing, feature engineering, and model optimization (Logistic Regression, Random Forest, XGBoost), prioritizing Recall and ROC-AUC performance.

[Credit Card Fraud Detection .ipynb](https://github.com/user-attachments/files/23568669/Credit.Card.Fraud.Detection.ipynb)

## Dataset

The dataset includes anonymized credit card transactions with:

- **Class 1:** Fraudulent  
- **Class 0:** Legitimate  

### Key Characteristics
- Features: Time, Amount, and **28 PCA-transformed features (V1–V28)**
- Extreme Imbalance: Fraud cases make up a very small percentage

##  Methodology

### 1. Data Processing

#### Preprocessing
- Scaled the `Amount` feature using **StandardScaler**
- Engineered `Transaction_Hour` from the `Time` column
- Dropped original `Time` and `Amount` after transformation

#### Analysis & Feature Engineering
- Explored class imbalance and fraud timing patterns
- Applied `np.log1p()` on scaled amount
- Created a **High_Value_Flag** for transactions above the 99th percentile

## Modeling & Evaluation

### Train–Test Split
- 80/20 split with **Stratified Sampling**

### Machine Learning Models
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  
*(All models configured to handle class imbalance)*

### Evaluation Metrics
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- ROC-AUC Score  

**Priority:** Accuracy on the minority class (fraud transactions)

## Results & Insights

### 1. Model Performance Summary

| Model | Fraud Recall | Fraud Precision | Fraud F1 | ROC-AUC | Insight |
|-------|--------------|-----------------|----------|----------|---------|
| Logistic Regression | 0.92 | 0.06 | 0.11 | 0.971 | Very high Recall but too many False Positives |
| Random Forest | 0.77 | 0.96 | 0.85 | 0.953 | Best Precision → highly reliable alerts |
| XGBoost | 0.83 | 0.89 | 0.86 | 0.973 | Best balance → top F1 & AUC |


### 2. Confusion Matrix Breakdown

| Model | TP | FN | FP | TN | Summary |
|--------|----|-----|-----|------|---------|
| Logistic Regression | 908 | 81 | 1,410 | 55,454 | High Recall but too many false alarms |
| Random Forest | 752 | 33 | **3** | 56,861 | Most reliable for production alerts |
| XGBoost | 811 | 7 | 10 | 56,854 | Balanced & best-performing model overall |


## Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost
