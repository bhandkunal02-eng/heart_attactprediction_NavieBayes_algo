import numpy as np
import pandas as pd
import pytest

from model import (
    DATA_PATH,
    CODE_LABELS,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    DatasetError,
    InputValidationError,
    load_dataset,
    predict_record,
    train_model,
    validate_input,
)


@pytest.fixture(scope="module")
def artifacts():
    return train_model()


def test_dataset_schema_and_rows():
    data = load_dataset(DATA_PATH)
    assert tuple(data.columns) == EXPECTED_COLUMNS
    assert len(data) == 303
    assert set(data["target"].unique()) == {0, 1}
    assert not data.isna().any().any()


def test_notebook_holdout_metrics_are_reproduced(artifacts):
    assert artifacts.metrics["train_rows"] == 242
    assert artifacts.metrics["test_rows"] == 61
    assert artifacts.metrics["accuracy"] == pytest.approx(0.8688524590163934)
    assert artifacts.metrics["precision"] == pytest.approx(0.9)
    assert artifacts.metrics["recall"] == pytest.approx(0.84375)
    assert 0 <= artifacts.metrics["roc_auc"] <= 1
    assert artifacts.confusion.sum() == 61


def test_model_and_feature_order_match_notebook(artifacts):
    assert artifacts.model.__class__.__name__ == "GaussianNB"
    assert tuple(artifacts.model.feature_names_in_) == FEATURE_COLUMNS
    assert artifacts.model.n_features_in_ == 13


def test_documented_sex_and_chest_pain_labels_match_source_codes(artifacts):
    assert CODE_LABELS["sex"] == {0: "Female", 1: "Male"}
    assert CODE_LABELS["cp"] == {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Asymptomatic",
    }
    assert set(artifacts.feature_values["sex"]) == {0, 1}
    assert set(artifacts.feature_values["cp"]) == {0, 1, 2, 3}


def test_prediction_uses_model_native_probabilities(artifacts):
    row = artifacts.dataset.iloc[0]
    values = {column: float(row[column]) for column in FEATURE_COLUMNS}
    prediction, probabilities = predict_record(values, artifacts)
    assert prediction in (0, 1)
    assert set(probabilities) == {0, 1}
    assert sum(probabilities.values()) == pytest.approx(1)
    assert np.isfinite(list(probabilities.values())).all()


def test_input_is_returned_in_canonical_column_order(artifacts):
    row = artifacts.dataset.iloc[0]
    values = {column: float(row[column]) for column in reversed(FEATURE_COLUMNS)}
    assert tuple(validate_input(values, artifacts).columns) == FEATURE_COLUMNS


@pytest.mark.parametrize(
    "update, message",
    [
        ({"age": float("nan")}, "finite"),
        ({"age": 1000}, "observed dataset range"),
        ({"sex": 99}, "observed dataset range"),
        ({"sex": 0.5}, "code present"),
        ({"unknown": 1}, "schema mismatch"),
    ],
)
def test_invalid_prediction_inputs_are_rejected(artifacts, update, message):
    row = artifacts.dataset.iloc[0]
    values = {column: float(row[column]) for column in FEATURE_COLUMNS}
    values.update(update)
    with pytest.raises(InputValidationError, match=message):
        validate_input(values, artifacts)


def test_loader_rejects_wrong_schema(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"wrong": [1], "target": [0]}).to_csv(path, index=False)
    with pytest.raises(DatasetError, match="columns/order"):
        load_dataset(path)


def test_loader_rejects_missing_values(tmp_path):
    data = pd.DataFrame([[1] * len(EXPECTED_COLUMNS)], columns=EXPECTED_COLUMNS)
    data.loc[0, "target"] = np.nan
    path = tmp_path / "missing.csv"
    data.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="missing values"):
        load_dataset(path)
