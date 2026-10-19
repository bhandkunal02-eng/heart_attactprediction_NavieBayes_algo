"""Training, validation, and prediction helpers for CardioGuard AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

DATA_PATH = Path(__file__).resolve().parent / "heart.csv"
FEATURE_COLUMNS = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
)
TARGET_COLUMN = "target"
EXPECTED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
TARGET_LABELS = (0, 1)
CONTINUOUS_FEATURES = ("age", "trestbps", "chol", "thalach", "oldpeak")
CODE_LABELS = {
    "sex": {
        0: "Female",
        1: "Male",
    },
    "cp": {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Asymptomatic",
    },
}

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest-pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar code",
    "restecg": "Resting ECG code",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina code",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST-segment slope code",
    "ca": "Major-vessel count code",
    "thal": "Thalassemia code",
}


class DatasetError(ValueError):
    """Raised when the source dataset does not match the model contract."""


class InputValidationError(ValueError):
    """Raised when prediction input does not match the training schema."""


@dataclass(frozen=True)
class ModelArtifacts:
    model: GaussianNB
    dataset: pd.DataFrame
    metrics: dict[str, Any]
    y_test: pd.Series
    y_pred: np.ndarray
    y_probability: np.ndarray
    confusion: np.ndarray
    feature_ranges: dict[str, tuple[float, float]]
    feature_values: dict[str, tuple[float, ...]]


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the original CSV and validate its schema and values."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DatasetError(f"Dataset file was not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read dataset file {csv_path}: {exc}") from exc
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise DatasetError(
            f"Dataset columns/order differ from the notebook schema. "
            f"Expected {list(EXPECTED_COLUMNS)}, found {list(frame.columns)}."
        )
    if frame.empty:
        raise DatasetError("Dataset has no rows.")
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise DatasetError(f"Dataset contains missing values in: {missing}.")
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"Dataset contains non-numeric values: {exc}") from exc
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise DatasetError("Dataset contains non-finite values.")
    labels = set(frame[TARGET_COLUMN].unique().tolist())
    if labels != set(TARGET_LABELS):
        raise DatasetError(f"Expected target labels {list(TARGET_LABELS)}, found {sorted(labels)}.")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def train_model(path: str | Path = DATA_PATH) -> ModelArtifacts:
    """Reproduce the notebook's GaussianNB training and 80/20 holdout."""
    dataset = load_dataset(path)
    X = dataset.loc[:, FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    positive_index = list(model.classes_).index(1)
    y_probability = model.predict_proba(X_test)[:, positive_index]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_rows": len(dataset),
        "feature_count": len(FEATURE_COLUMNS),
        "class_counts": {
            int(label): int(count)
            for label, count in dataset[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }
    return ModelArtifacts(
        model=model,
        dataset=dataset,
        metrics=metrics,
        y_test=y_test.reset_index(drop=True),
        y_pred=np.asarray(y_pred),
        y_probability=np.asarray(y_probability),
        confusion=confusion_matrix(y_test, y_pred, labels=TARGET_LABELS),
        feature_ranges={
            column: (float(dataset[column].min()), float(dataset[column].max()))
            for column in FEATURE_COLUMNS
        },
        feature_values={
            column: tuple(float(value) for value in sorted(dataset[column].unique()))
            for column in FEATURE_COLUMNS
        },
    )


def validate_input(
    values: dict[str, Any] | pd.DataFrame, artifacts: ModelArtifacts
) -> pd.DataFrame:
    """Validate one record and return features in canonical order."""
    if isinstance(values, pd.DataFrame):
        if len(values) != 1:
            raise InputValidationError("Provide exactly one record for prediction.")
        frame = values.copy()
    elif isinstance(values, dict):
        frame = pd.DataFrame([values])
    else:
        raise InputValidationError("Prediction input must be a mapping or one-row DataFrame.")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in FEATURE_COLUMNS]
    if missing or extra:
        raise InputValidationError(
            f"Feature schema mismatch. Missing: {missing or 'none'}; "
            f"unexpected: {extra or 'none'}."
        )
    try:
        frame = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"All features must be numeric: {exc}") from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise InputValidationError("All features must be finite numbers.")
    for column in FEATURE_COLUMNS:
        value = float(frame.iloc[0][column])
        lower, upper = artifacts.feature_ranges[column]
        if value < lower or value > upper:
            raise InputValidationError(
                f"{FEATURE_LABELS[column]} must be within the observed dataset range "
                f"{lower:g}–{upper:g}."
            )
        if column not in CONTINUOUS_FEATURES and value not in artifacts.feature_values[column]:
            codes = ", ".join(f"{item:g}" for item in artifacts.feature_values[column])
            raise InputValidationError(
                f"{FEATURE_LABELS[column]} must use a code present in the dataset: {codes}."
            )
    return frame


def predict_record(
    values: dict[str, Any] | pd.DataFrame, artifacts: ModelArtifacts
) -> tuple[int, dict[int, float]]:
    """Return the model's class and native probabilities."""
    record = validate_input(values, artifacts)
    prediction = int(artifacts.model.predict(record)[0])
    probabilities = artifacts.model.predict_proba(record)[0]
    return prediction, {
        int(label): float(probability)
        for label, probability in zip(artifacts.model.classes_, probabilities)
    }
