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

ROOT = Path(__file__).resolve().parent
(ROOT / "model").mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)
raw = load_breast_cancer(as_frame=True).frame.copy()
raw["target"] = raw["target"].map({0: "malignant", 1: "benign"})
X, y = raw.drop(columns="target"), raw["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
models = {
    "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
    "Decision Tree": Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", DecisionTreeClassifier(random_state=42))]),
    "K-Nearest Neighbors": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", GaussianNB())]),
    "Support Vector Machine": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", SVC(probability=True, random_state=42))])
}
rows = []
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    joblib.dump(estimator, ROOT / "model" / (name.lower().replace(" ", "_").replace("-", "") + ".joblib"))
    predicted = estimator.predict(X_test)
    classes = list(estimator.named_steps["classifier"].classes_)
    p_malignant = estimator.predict_proba(X_test)[:, classes.index("malignant")]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score((y_test == "malignant").astype(int), p_malignant),
        "Precision": precision_score(y_test, predicted, pos_label="malignant", zero_division=0),
        "Recall": recall_score(y_test, predicted, pos_label="malignant", zero_division=0),
        "F1": f1_score(y_test, predicted, pos_label="malignant", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted)
    })
pd.DataFrame(rows).to_csv(ROOT / "reports" / "metrics.csv", index=False)
# Store the held-out test data with target column, as required by the assignment.
test = X_test.copy()
test["target"] = y_test.values
test.to_csv(ROOT / "test_data.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))