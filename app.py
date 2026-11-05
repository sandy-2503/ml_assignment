import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

APP_DIRECTORY = Path(__file__).resolve().parent
TARGET_COLUMN = "target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_LABELS = ["malignant", "benign"]

@st.cache_resource
def load_trained_models():
    return {
        model_name: joblib.load(APP_DIRECTORY / "model" / model_filename)
        for model_name, model_filename in MODEL_FILES.items()
    }

st.set_page_config(page_title="AIML Classification Model Lab", page_icon="🧪", layout="wide")
st.title("🧪 Classification Model Evaluation Lab")
st.write(
    "Compare five pre-trained classifiers on held-out test data from the "
    "Wisconsin Diagnostic Breast Cancer dataset. The app does not retrain models."
)
st.info("Upload a CSV containing all 30 model input features and a `target` column with values `malignant` or `benign`.")
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.caption("For a quick demo, upload the repository's `test_data.csv` file.")
    st.stop()

try:
    test_data = pd.read_csv(uploaded)
except Exception as error:
    st.error(f"Could not read CSV: {error}")
    st.stop()

if TARGET_COLUMN not in test_data.columns:
    st.error("The uploaded CSV must contain the target column named `target`.")
    st.stop()
if test_data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

actual_labels = test_data[TARGET_COLUMN].astype(str).str.strip().str.lower()
if not set(actual_labels.unique()).issubset(set(CLASS_LABELS)):
    st.error("Target values must be `malignant` or `benign`.")
    st.stop()

trained_models = load_trained_models()
required_features = list(trained_models["Logistic Regression"].feature_names_in_)
missing_features = [
    feature_name for feature_name in required_features
    if feature_name not in test_data.columns
]
if missing_features:
    st.error("Missing required input feature columns: " + ", ".join(missing_features))
    st.stop()
test_features = test_data[required_features].apply(pd.to_numeric, errors="coerce")
if test_features.isna().all(axis=None):
    st.error("No numeric feature values were found. Check the uploaded CSV.")
    st.stop()

st.subheader("Uploaded test data")
st.write(f"Rows: **{len(test_data)}** · Input features: **{len(required_features)}**")
st.dataframe(test_data.head(10), use_container_width=True)

selected_model_name = st.selectbox("Choose a model to inspect", list(MODEL_FILES.keys()))
metric_rows = []
predictions_by_model = {}
for model_name, model in trained_models.items():
    predicted_labels = model.predict(test_features)
    model_classes = list(model.named_steps["classifier"].classes_)
    malignant_probabilities = model.predict_proba(test_features)[
        :, model_classes.index("malignant")
    ]
    auc_score = (
        roc_auc_score((actual_labels == "malignant").astype(int), malignant_probabilities)
        if actual_labels.nunique() == 2
        else float("nan")
    )
    metric_rows.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(actual_labels, predicted_labels),
        "AUC": auc_score,
        "Precision": precision_score(
            actual_labels, predicted_labels, pos_label="malignant", zero_division=0
        ),
        "Recall": recall_score(
            actual_labels, predicted_labels, pos_label="malignant", zero_division=0
        ),
        "F1": f1_score(
            actual_labels, predicted_labels, pos_label="malignant", zero_division=0
        ),
        "MCC": matthews_corrcoef(actual_labels, predicted_labels),
    })
    predictions_by_model[model_name] = predicted_labels

st.subheader("Comparison of all five models")
comparison = pd.DataFrame(metric_rows).set_index("ML Model Name")
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
st.bar_chart(comparison[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]])

st.subheader(f"Detailed results: {selected_model_name}")
selected_predictions = predictions_by_model[selected_model_name]
selected_metrics = comparison.loc[selected_model_name]
metric_columns = st.columns(6)
for metric_column, metric_name in zip(
    metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
):
    metric_column.metric(
        metric_name,
        f"{selected_metrics[metric_name]:.4f}"
        if pd.notna(selected_metrics[metric_name])
        else "N/A",
    )

st.write("Confusion matrix (rows = actual, columns = predicted; label order: malignant, benign)")
confusion_matrix_values = confusion_matrix(
    actual_labels, selected_predictions, labels=CLASS_LABELS
)
st.dataframe(pd.DataFrame(
    confusion_matrix_values,
    index=[f"Actual {label}" for label in CLASS_LABELS],
    columns=[f"Predicted {label}" for label in CLASS_LABELS],
))
st.text("Classification report")
st.code(classification_report(
    actual_labels,
    selected_predictions,
    labels=CLASS_LABELS,
    target_names=CLASS_LABELS,
    zero_division=0,
))

st.download_button(
    "Download comparison metrics as CSV",
    data=comparison.reset_index().to_csv(index=False).encode("utf-8"),
    file_name="model_comparison_metrics.csv",
    mime="text/csv",
)
