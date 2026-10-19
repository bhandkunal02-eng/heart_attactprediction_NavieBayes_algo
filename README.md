# CardioGuard AI

<p align="center">
  <img src="assets/cardioguard-logo.svg" alt="CardioGuard AI logo" width="132">
</p>

<p align="center">
  <a href="https://bhandkunal02-eng-heart-attactprediction-naviebayes--app-hcqkfp.streamlit.app/"><strong>Open the live CardioGuard AI app</strong></a>
</p>

An educational, interactive heart-disease classification project built around the repository's existing dataset and Naive Bayes notebook. The Streamlit dashboard demonstrates real model predictions, reproducible evaluation, aggregate data visualization, and responsible-AI limitations.

> **Educational demonstration only — not a medical device or medical advice.** It cannot diagnose heart disease or estimate your personal risk. If you may be having a heart attack, call your local emergency number immediately. Do not wait for this app or a model prediction.

## What the source project actually contains

- `heart.csv`: 303 records, 13 features, and the binary `target` values 0 and 1. Rows are never shown in the explorer; it displays aggregates only.
- `Naive_Bayed_implementaion.ipynb`: fits scikit-learn `GaussianNB` directly on the numeric feature values.
- The notebook uses `train_test_split(test_size=0.2, random_state=42)` with no stratification. It performs no imputation, scaling, or other feature preprocessing.
- The notebook's saved holdout outputs are accuracy **0.8689**, precision **0.9000**, and recall **0.8438**. The app independently recalculates these metrics from the same split and model code. ROC-AUC and confusion matrix are also calculated from that held-out set.
- There was no serialized estimator in the source repository. On first app load, the app trains the notebook's model from the checked-in CSV and caches it for the Streamlit process. It does not use a dummy model or silently switch to another algorithm.

### Input schema

The app uses the CSV/notebook feature order exactly:

| Column | Meaning documented by the dataset | Input representation |
|---|---|---|
| `age` | Age | Numeric |
| `sex` | Sex | `0` = Female; `1` = Male |
| `cp` | Chest-pain type | `0` = Typical angina; `1` = Atypical angina; `2` = Non-anginal pain; `3` = Asymptomatic |
| `trestbps` | Resting blood pressure | Numeric |
| `chol` | Serum cholesterol | Numeric |
| `fbs` | Fasting blood sugar code | Numeric code |
| `restecg` | Resting ECG result | Numeric code |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise-induced angina | Numeric code |
| `oldpeak` | ST depression | Numeric |
| `slope` | ST-segment slope | Numeric code |
| `ca` | Major-vessel count code | Numeric code |
| `thal` | Thalassemia code | Numeric code |

The source dataset uses the standard UCI Cleveland heart-disease sex codes and chest-pain categories, with the chest-pain values shifted down by one from the codebook's 1-4 notation. The prediction form shows the words alongside the unchanged CSV codes so the exact numeric values still go into the model. Other category values and target labels remain numeric because their meanings are not documented in this repository. Prediction input validation checks the expected feature set, finite numbers, dataset-observed ranges, and observed category codes.

## Dashboard

- **Overview:** model and dataset summary, with the medical-use warning.
- **Try the model:** validated form calling the trained `GaussianNB` estimator and displaying its native class probabilities (not calibrated personal risk).
- **Model analytics:** held-out accuracy, precision, recall, ROC-AUC, confusion matrix, and ROC curve, labeled as test-split results.
- **Dataset explorer:** class counts, individual feature distributions, and aggregate summary statistics only.
- **About & limitations:** schema, method, and limitations.

## Run locally

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/bhandkunal02-eng/heart_attactprediction_NavieBayes_algo.git
cd heart_attactprediction_NavieBayes_algo
python -m venv .venv
```

Activate the environment, then install and launch:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local address (usually `http://localhost:8501`). The model and dataset are local to the app process; prediction form inputs are not saved by this application.

## Run tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests verify the CSV schema and row count, the exact model/split and saved notebook metrics, prediction probabilities, feature ordering, and invalid-input handling.

## Deploy on Streamlit Community Cloud

**Live app:** [CardioGuard AI](https://bhandkunal02-eng-heart-attactprediction-naviebayes--app-hcqkfp.streamlit.app/)
It is currently deployed from the `bhandkunal02-eng-cardioguard-ai` branch with `app.py` as its entry point.

1. Push this project to a GitHub repository that Streamlit Community Cloud can access. Keep `heart.csv` in the repository root; do not upload private health data or secrets.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and select **Create app**.
3. Select the repository and branch, set the **Main file path** to `app.py`, and deploy. Community Cloud installs dependencies from `requirements.txt`.
4. Open the generated app URL and verify the overview, analytics, explorer, and prediction pages.

For future updates, push commits to the deployed branch; Community Cloud can redeploy from the repository. Do not commit `.streamlit/secrets.toml`, credentials, or private data. The `.gitignore` excludes common local secrets and generated model artifacts.

### Push this worktree branch

```bash
git push -u origin bhandkunal02-eng-cardioguard-ai
```

Push subsequent changes to the deployed branch to trigger a Streamlit Community Cloud rebuild.

## Project files

```text
app.py                         Streamlit dashboard and visualizations
model.py                       Dataset contract, model training, validation, predictions
heart.csv                      Source dataset used by the notebook and app
Naive_Bayed_implementaion.ipynb Original model notebook
tests/test_model.py            Model, data, and input validation tests
requirements.txt               Runtime dependencies
requirements-dev.txt           Runtime plus test dependencies
```

## Responsible-use notes

This application is for learning and portfolio demonstration only. The dataset is small and historical, a single fixed split is not external validation, and Naive Bayes assumptions may not fit the data. The model output and probabilities are not a clinical assessment. Do not use them to make care decisions or delay medical attention.
