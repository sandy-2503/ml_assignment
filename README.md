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
Add your repository URL here after creating and pushing the project: **[https://github.com/sandy-2503/ml_assignment]**

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

### Observations on model performance
| ML Model Name | Observation |
|---|---|
| Logistic Regression | Accuracy=0.9650; AUC=0.9962; Precision=0.9800; Recall=0.9245; F1=0.9515; MCC=0.9251. Interpret metrics together; higher values generally indicate better performance. |
| Decision Tree | Accuracy=0.9580; AUC=0.9473; Precision=0.9796; Recall=0.9057; F1=0.9412; MCC=0.9103. Interpret metrics together; higher values generally indicate better performance. |
| K-Nearest Neighbors | Accuracy=0.9580; AUC=0.9860; Precision=0.9796; Recall=0.9057; F1=0.9412; MCC=0.9103. Interpret metrics together; higher values generally indicate better performance. |
| Naive Bayes | Accuracy=0.9441; AUC=0.9952; Precision=1.0000; Recall=0.8491; F1=0.9184; MCC=0.8830. Interpret metrics together; higher values generally indicate better performance. |
| Support Vector Machine | Accuracy=0.9720; AUC=0.9966; Precision=1.0000; Recall=0.9245; F1=0.9608; MCC=0.9408. Interpret metrics together; higher values generally indicate better performance. |
| Overall winner for this dataset | Based on the highest AUC, with F1 and MCC used as tie-breakers, **Support Vector Machine** is the winner in this run. The choice should also consider the cost of false negatives for the malignant class. |

## Running locally
1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Upload `test_data.csv` in the app.

## Repository contents
- `app.py`: Streamlit UI and evaluation workflow.
- `train_models.py`: Reproducible training and metric-generation script.
- `requirements.txt`: Python dependencies.
- `test_data.csv`: Held-out test records with features and target.
- `model/`: Five saved model pipelines.
- `reports/metrics.csv`: Metric comparison table.
- `reports/evaluation_details.json`: Confusion matrices and classification reports from the generated test run.
