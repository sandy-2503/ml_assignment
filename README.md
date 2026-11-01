# Classification Model Evaluation Lab

## a. Problem statement
Implement and compare five classification algorithms on one classification dataset, calculate the six required evaluation metrics, and provide an interactive Streamlit app that evaluates uploaded test data without retraining.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), distributed through `scikit-learn` from the UCI Machine Learning Repository.
- **Task:** Binary classification of breast tumour observations into `malignant` and `benign`.
- **Size:** 569 instances and 30 numeric input features (meets the assignment minimum of 500 instances and 12 features).
- **Target column:** `target` (`malignant` / `benign`).
- **Split:** 75% training and 25% held-out test data, stratified, `random_state=42`.
- **Positive class for Precision/Recall/F1 and AUC:** `malignant`.
- **Caution:** This is an educational ML exercise, not a medical diagnostic tool.
-