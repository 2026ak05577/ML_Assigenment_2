# Dry Bean Classification Lab

## a. Problem statement

Train and compare five multiclass classification models for the Dry Bean dataset. Evaluate all models on the same stratified holdout set using accuracy, macro one-vs-rest AUC, macro precision, macro recall, macro F1, and Matthews correlation coefficient (MCC). The Streamlit app evaluates labeled CSV data using saved models; it does not retrain on uploaded data.

## b. Dataset description

The [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) contains 13,611 instances, 16 numeric input features, and a seven-class target named `Class`. The classes identify bean varieties. The training script fetches the dataset using the UCI Machine Learning Repository API (`ucimlrepo`, dataset ID 602).

The experiment uses a stratified 75/25 train/test split with random seed 37. Scaling is applied to Logistic Regression, K-Nearest Neighbors, and Support Vector Machine inside each saved pipeline. The Decision Tree and Gaussian Naive Bayes models use the original feature scale. `test_data.csv` contains the held-out input columns and `Class` labels.

## c. GitHub Repository Link

**Repository:** TODO - replace this text with your public GitHub repository URL after pushing the project.

## d. Models used and evaluation

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9239 | 0.9948 | 0.9365 | 0.9344 | 0.9353 | 0.9080 |
| Decision Tree | 0.8869 | 0.9404 | 0.9003 | 0.9007 | 0.9004 | 0.8633 |
| KNN | 0.9218 | 0.9834 | 0.9368 | 0.9324 | 0.9342 | 0.9055 |
| Naive Bayes | 0.7728 | 0.9669 | 0.7807 | 0.7785 | 0.7768 | 0.7262 |
| Support Vector Machine | 0.9315 | 0.9953 | 0.9446 | 0.9403 | 0.9421 | 0.9172 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, balanced baseline. It achieved 0.9239 accuracy and 0.9353 macro F1, with a high 0.9948 macro OVR AUC. |
| Decision Tree | Easiest to interpret, but it had the lowest accuracy and MCC among the five models on this split. |
| KNN | Competitive with Logistic Regression in accuracy and macro F1, though its AUC was lower at 0.9834. |
| Naive Bayes | Fast baseline, but it had the lowest accuracy, precision, recall, F1, and MCC in this comparison. |
| Support Vector Machine | Best overall on this holdout: it led accuracy, AUC, macro precision, macro recall, macro F1, and MCC. |
| Overall winner | Support Vector Machine, based on the six reported metrics for this split. |

These are results from one fixed holdout split, not a guarantee of performance on other data. AUC is macro one-vs-rest; precision, recall, and F1 use macro averaging.

## Project files

```text
app.py
train_models.py
requirements.txt
README.md
test_data.csv
model/
  decision_tree.joblib
  gaussian_naive_bayes.joblib
  knearest_neighbors.joblib
  logistic_regression.joblib
  metrics.json
  support_vector_machine.joblib
```

## Run locally

Python 3.11 or newer is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`train_models.py` needs internet access to fetch the source dataset. It creates the five model artifacts, `model/metrics.json`, and the labeled held-out `test_data.csv`. The app starts with that bundled test CSV; a user can instead upload a CSV containing all 16 feature columns and the `Class` target column. The app displays metrics and a confusion matrix and allows downloading predictions. Uploaded rows are never used to fit a model.

## Deploy on Streamlit Community Cloud

1. Push this folder, including the `model/` artifacts, `test_data.csv`, `requirements.txt`, and this README, to your GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Wait for the app to finish building, then verify the public URL in a private browser window.
4. Replace the repository placeholder above and add the live app URL here: **TODO - paste your deployed Streamlit Community Cloud URL.**

## BITS Virtual Lab evidence and submission PDF

Run the experiment in BITS Virtual Lab and capture your own screenshot there. Include one screenshot in the PDF, along with this README content, the working GitHub repository URL, and the working Streamlit app URL in the assignment's required order. The screenshot and the two public URLs must be supplied by the student; they cannot be generated from this workspace.
