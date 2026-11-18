"""
Pytest suite for CSC418 SP23-BAI-031 pipeline.
Run after train.py has produced artifacts/.
"""
import json
import pandas as pd
import pytest
from pathlib import Path

ARTIFACTS = Path("artifacts")
STUDENT_ID = "SP23-BAI-031"
FEATURES = ["basket_volatility", "stock_gap", "promotion_load", "delivery_delay"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def metrics():
    path = ARTIFACTS / "metrics.json"
    assert path.exists(), "artifacts/metrics.json not found – run train.py first"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def bundle():
    import joblib
    path = ARTIFACTS / "model.joblib"
    assert path.exists(), "artifacts/model.joblib not found – run train.py first"
    return joblib.load(path)


# ── Artifact tests ────────────────────────────────────────────────────────────

class TestArtifacts:
    def test_metrics_file_exists(self):
        assert (ARTIFACTS / "metrics.json").exists()

    def test_model_file_exists(self):
        assert (ARTIFACTS / "model.joblib").exists()

    def test_student_id(self, metrics):
        assert metrics["student_id"] == STUDENT_ID, \
            f"Expected {STUDENT_ID}, got {metrics['student_id']}"

    def test_accuracy_in_range(self, metrics):
        acc = metrics["accuracy"]
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of [0, 1]"

    def test_accuracy_above_baseline(self, metrics):
        """Model must beat a 50% random baseline."""
        assert metrics["accuracy"] >= 0.50, \
            f"Accuracy {metrics['accuracy']} is below 0.50 baseline"

    def test_training_rows_minimum(self, metrics):
        assert metrics["training_rows"] >= 100, \
            f"Only {metrics['training_rows']} rows – need >= 100"

    def test_model_version_present(self, metrics):
        assert "model_version" in metrics and len(metrics["model_version"]) > 0

    def test_dataset_sha256_present(self, metrics):
        assert "dataset_sha256" in metrics and len(metrics["dataset_sha256"]) == 64


# ── Bundle / model tests ──────────────────────────────────────────────────────

class TestBundle:
    def test_bundle_has_model_key(self, bundle):
        assert "model" in bundle

    def test_bundle_has_features_key(self, bundle):
        assert "features" in bundle

    def test_features_match_config(self, bundle):
        assert bundle["features"] == FEATURES, \
            f"Expected {FEATURES}, got {bundle['features']}"

    def test_predict_proba_shape(self, bundle):
        sample = pd.DataFrame([{
            "basket_volatility": 0.5,
            "stock_gap": -0.2,
            "promotion_load": 0.8,
            "delivery_delay": 0.1,
        }])
        proba = bundle["model"].predict_proba(sample)
        assert proba.shape == (1, 2), f"Unexpected shape: {proba.shape}"

    def test_predict_proba_sums_to_one(self, bundle):
        sample = pd.DataFrame([{
            "basket_volatility": 0.0,
            "stock_gap": 0.0,
            "promotion_load": 0.0,
            "delivery_delay": 0.0,
        }])
        proba = bundle["model"].predict_proba(sample)
        total = proba[0].sum()
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}"

    def test_handles_missing_values(self, bundle):
        """Model pipeline must handle NaN (imputer is in the pipeline)."""
        import numpy as np
        sample = pd.DataFrame([{
            "basket_volatility": float("nan"),
            "stock_gap": 0.3,
            "promotion_load": float("nan"),
            "delivery_delay": -0.1,
        }])
        proba = bundle["model"].predict_proba(sample)
        assert proba.shape == (1, 2)

    def test_batch_prediction(self, bundle):
        rows = [
            {"basket_volatility": i * 0.1, "stock_gap": -i * 0.1,
             "promotion_load": 0.5, "delivery_delay": 0.2}
            for i in range(10)
        ]
        df = pd.DataFrame(rows)
        preds = bundle["model"].predict(df)
        assert len(preds) == 10
        assert set(preds).issubset({0, 1}), f"Unexpected labels: {set(preds)}"


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestDataset:
    def test_dataset_exists(self):
        assert Path("dataset/train.csv").exists()

    def test_dataset_minimum_rows(self):
        df = pd.read_csv("dataset/train.csv")
        assert len(df) >= 100, f"Dataset has only {len(df)} rows"

    def test_dataset_has_required_columns(self):
        df = pd.read_csv("dataset/train.csv")
        required = set(FEATURES + ["label"])
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_label_binary(self):
        df = pd.read_csv("dataset/train.csv")
        assert df["label"].nunique() == 2, \
            f"Expected 2 label classes, got {df['label'].nunique()}"
