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

## c. GitHub Repository Link
Add your repository URL here after creating and pushing the project: **[PASTE YOUR GITHUB REPOSITORY URL]**

## d. Models used

All models use the same training/test split. Scaling is used for Logistic Regression, KNN, and SVM. Missing values are median-imputed within each pipeline. The saved models are in `model/`.

### Evaluation metrics
| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.965  | 0.9962 |      0.98   |   0.9245 | 0.9515 | 0.9251 |
| Decision Tree          |     0.958  | 0.9473 |      0.9796 |   0.9057 | 0.9412 | 0.9103 |
| K-Nearest Neighbors    |     0.958  | 0.986  |      0.9796 |   0.9057 | 0.9412 | 0.9103 |
| Naive Bayes            |     0.9441 | 0.9952 |      1      |   0.8491 | 0.9184 | 0.883  |
| Support Vector Machine |     0.972  | 0.9966 |      1      |   0.9245 | 0.9608 | 0.9408 |

Metrics are calculated on the held-out test set. AUC uses the predicted probability for the malignant class. Precision, recall, and F1 treat malignant as the positive class.