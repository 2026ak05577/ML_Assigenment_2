from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Model Lab", page_icon="DB", layout="wide")
st.title("Dry Bean Classification Lab")
st.caption("Seven bean classes · 16 measured features · fixed holdout evaluation")


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


model_name = st.selectbox("Classifier", list(MODEL_FILES))
uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
if uploaded_file is None:
    data_path = ROOT / "test_data.csv"
    if not data_path.exists():
        st.error("The bundled test_data.csv is missing. Run train_models.py first.")
        st.stop()
    test_data = pd.read_csv(data_path)
    st.info("Showing the bundled held-out test set. Upload another labeled CSV to evaluate it.")
else:
    try:
        test_data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

model_path = MODEL_DIR / MODEL_FILES[model_name]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}. Run train_models.py first.")
    st.stop()
estimator = load_model(str(model_path))
feature_columns = list(estimator.feature_names_in_)
missing_columns = [column for column in [*feature_columns, TARGET_COLUMN] if column not in test_data]
if missing_columns:
    st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
    st.stop()

X_test = test_data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
y_test = test_data[TARGET_COLUMN]
if X_test.isna().any().any() or y_test.isna().any():
    st.error("Required columns must contain complete, valid values.")
    st.stop()

class_labels = list(estimator.classes_)
unknown_labels = sorted(set(y_test) - set(class_labels))
if unknown_labels:
    st.error(f"Target values are not recognized by the saved model: {unknown_labels}")
    st.stop()

predictions = estimator.predict(X_test)
probabilities = estimator.predict_proba(X_test)
try:
    auc = roc_auc_score(
        y_test,
        probabilities,
        labels=class_labels,
        multi_class="ovr",
        average="macro",
    )
except ValueError:
    auc = None

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC (macro OVR)": auc,
    "Precision (macro)": precision_score(
        y_test, predictions, labels=class_labels, average="macro", zero_division=0
    ),
    "Recall (macro)": recall_score(
        y_test, predictions, labels=class_labels, average="macro", zero_division=0
    ),
    "F1 (macro)": f1_score(
        y_test, predictions, labels=class_labels, average="macro", zero_division=0
    ),
    "MCC": matthews_corrcoef(y_test, predictions),
}

st.subheader(f"{model_name} · {len(test_data):,} test rows")
metric_columns = st.columns(len(metrics))
for column, (label, value) in zip(metric_columns, metrics.items()):
    column.metric(label, "N/A" if value is None else f"{value:.4f}")
if auc is None:
    st.warning("AUC requires test rows from every class; the other metrics are available.")

st.subheader("Confusion matrix")
confusion = confusion_matrix(y_test, predictions, labels=class_labels)
st.dataframe(
    pd.DataFrame(confusion, index=class_labels, columns=class_labels),
    width="stretch",
)

results = test_data.copy()
results["Predicted Class"] = predictions
st.download_button(
    "Download predictions",
    results.to_csv(index=False).encode("utf-8"),
    file_name="dry_bean_predictions.csv",
    mime="text/csv",
)
