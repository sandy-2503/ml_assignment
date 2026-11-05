"""
Train five classification models on the Wisconsin Diagnostic Breast Cancer dataset.
Dataset: scikit-learn's copy of the UCI Wisconsin Diagnostic Breast Cancer dataset.
The dataset contains 569 instances and 30 numeric features.
"""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

PROJECT_DIRECTORY = Path(__file__).resolve().parent
(PROJECT_DIRECTORY / "model").mkdir(exist_ok=True)
(PROJECT_DIRECTORY / "reports").mkdir(exist_ok=True)
dataset = load_breast_cancer(as_frame=True).frame.copy()
dataset["target"] = dataset["target"].map({0: "malignant", 1: "benign"})
features = dataset.drop(columns="target")
labels = dataset["target"]
training_features, testing_features, training_labels, testing_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42, stratify=labels
)
classifiers = {
    "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
    "Decision Tree": Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", DecisionTreeClassifier(random_state=42))]),
    "K-Nearest Neighbors": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", GaussianNB())]),
    "Support Vector Machine": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", SVC(probability=True, random_state=42))])
}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
metric_rows = []
for classifier_name, classifier_pipeline in classifiers.items():
    classifier_pipeline.fit(training_features, training_labels)
    joblib.dump(
        classifier_pipeline,
        PROJECT_DIRECTORY / "model" / MODEL_FILES[classifier_name],
    )
    predicted_labels = classifier_pipeline.predict(testing_features)
    classifier_classes = list(
        classifier_pipeline.named_steps["classifier"].classes_
    )
    malignant_probabilities = classifier_pipeline.predict_proba(testing_features)[
        :, classifier_classes.index("malignant")
    ]
    metric_rows.append({
        "ML Model Name": classifier_name,
        "Accuracy": accuracy_score(testing_labels, predicted_labels),
        "AUC": roc_auc_score(
            (testing_labels == "malignant").astype(int), malignant_probabilities
        ),
        "Precision": precision_score(
            testing_labels, predicted_labels, pos_label="malignant", zero_division=0
        ),
        "Recall": recall_score(
            testing_labels, predicted_labels, pos_label="malignant", zero_division=0
        ),
        "F1": f1_score(
            testing_labels, predicted_labels, pos_label="malignant", zero_division=0
        ),
        "MCC": matthews_corrcoef(testing_labels, predicted_labels)
    })
metrics_dataframe = pd.DataFrame(metric_rows)
metrics_dataframe.to_csv(
    PROJECT_DIRECTORY / "reports" / "metrics.csv", index=False
)
# Store the held-out test data with target column, as required by the assignment.
test_dataset = testing_features.copy()
test_dataset["target"] = testing_labels.values
test_dataset.to_csv(PROJECT_DIRECTORY / "test_data.csv", index=False)
print(metrics_dataframe.round(4).to_string(index=False))
