from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_SIZE = 0.25
RANDOM_STATE = 37


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=602)
    features = dataset.data.features.copy()
    target = dataset.data.targets.iloc[:, 0].copy()

    if features.shape[1] < 12 or len(features) < 500:
        raise ValueError("The selected dataset does not meet the assignment's minimum size.")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("The dataset contains missing values; add an explicit imputation step.")

    return features, target


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)
        ),
    }


def main() -> None:
    features, target = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    results: list[dict[str, float | str]] = []
    for name, estimator in build_models().items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)
        auc = roc_auc_score(
            y_test,
            probabilities,
            labels=estimator.classes_,
            multi_class="ovr",
            average="macro",
        )
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "auc": auc,
                "precision": precision_score(y_test, predictions, average="macro", zero_division=0),
                "recall": recall_score(y_test, predictions, average="macro", zero_division=0),
                "f1": f1_score(y_test, predictions, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y_test, predictions),
            }
        )
        filename = name.lower().replace(" ", "_").replace("-", "") + ".joblib"
        joblib.dump(estimator, MODEL_DIR / filename)

    pd.DataFrame(X_test.assign(Class=y_test)).to_csv(ROOT / "test_data.csv", index=False)
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    print(f"Dataset: {len(features)} rows, {features.shape[1]} features, {target.nunique()} classes")
    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved models and metrics in {MODEL_DIR}")
    print(f"Saved held-out rows to {ROOT / 'test_data.csv'}")


if __name__ == "__main__":
    main()
