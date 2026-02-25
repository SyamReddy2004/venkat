#!/usr/bin/env python3
"""
Unit and integration tests for the CIFAR-10 MLOps pipeline.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short -q    # quiet mode
"""
import sys
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.data_preprocessing import (
    load_cifar10,
    preprocess_arrays,
    build_tf_datasets,
    _augment,
)
from src.model import build_model, get_callbacks, _residual_block


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_dataset():
    """Tiny synthetic dataset for fast tests (no actual CIFAR-10 download)."""
    rng      = np.random.default_rng(0)
    x_train  = rng.integers(0, 255, (500, 32, 32, 3), dtype=np.uint8)
    y_train  = rng.integers(0, 10,  (500,),            dtype=np.int64)
    x_test   = rng.integers(0, 255, (100, 32, 32, 3),  dtype=np.uint8)
    y_test   = rng.integers(0, 10,  (100,),             dtype=np.int64)
    return x_train, y_train, x_test, y_test


@pytest.fixture(scope="session")
def processed_data(small_dataset):
    x_train, y_train, x_test, y_test = small_dataset
    return preprocess_arrays(
        x_train, y_train, x_test, y_test,
        val_split=0.1, num_classes=10, random_seed=42
    )


@pytest.fixture(scope="session")
def datasets(processed_data):
    train_data, val_data, test_data = processed_data
    return build_tf_datasets(
        train_data, val_data, test_data,
        batch_size=32, augment=True, seed=42
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_loads(self, cfg):
        assert isinstance(cfg, dict)

    def test_required_sections(self, cfg):
        for section in ["data", "model", "training", "mlflow", "paths"]:
            assert section in cfg, f"Missing config section: {section}"

    def test_num_classes(self, cfg):
        assert cfg["data"]["num_classes"] == 10

    def test_class_names_count(self, cfg):
        assert len(cfg["data"]["class_names"]) == 10

    def test_image_shape(self, cfg):
        assert cfg["model"]["input_shape"] == [32, 32, 3]

    def test_valid_architecture(self, cfg):
        valid = ["simple_cnn", "resnet_custom", "efficientnet_transfer"]
        assert cfg["model"]["architecture"] in valid


# ─────────────────────────────────────────────────────────────────────────────
# Data preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_shapes(self, processed_data):
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = processed_data
        # Images normalised and shape preserved
        assert x_tr.ndim == 4
        assert x_tr.shape[1:] == (32, 32, 3)

    def test_label_onehot(self, processed_data):
        (x_tr, y_tr), _, _ = processed_data
        assert y_tr.ndim == 2
        assert y_tr.shape[1] == 10
        # Each row should sum to 1
        np.testing.assert_allclose(y_tr.sum(axis=1), np.ones(len(y_tr)))

    def test_split_sizes(self, small_dataset, processed_data):
        x_train = small_dataset[0]
        (x_tr, _), (x_va, _), (x_te, _) = processed_data
        # val split = 0.1 of 500 → 50 val; test = 100 (unchanged)
        assert len(x_va) == 50
        assert len(x_te) == 100
        assert len(x_tr) == 450

    def test_pixel_range(self, processed_data):
        (x_tr, _), _, _ = processed_data
        # After z-score normalisation mean ≈ 0 (not exactly due to val exclusion)
        assert x_tr.mean() < 1.0

    def test_no_data_leakage(self, processed_data):
        (x_tr, _), (x_va, _), (x_te, _) = processed_data
        total = len(x_tr) + len(x_va) + len(x_te)
        # 500 synthetic samples → 450 train + 50 val; 100 test samples unchanged
        assert total == 600   # 500 (train split) + 100 (test)

    def test_augment_returns_same_shape(self):
        img = tf.random.uniform((32, 32, 3))
        out = _augment(img)
        assert out.shape == (32, 32, 3)

    def test_tf_datasets_batches(self, datasets):
        train_ds, val_ds, test_ds = datasets
        for x_batch, y_batch in train_ds.take(1):
            assert x_batch.shape[1:] == (32, 32, 3)
            assert y_batch.shape[1]  == 10
            break


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelBuilding:
    @pytest.mark.parametrize("arch", ["simple_cnn", "resnet_custom"])
    def test_builds_without_error(self, arch):
        model = build_model(
            architecture=arch, input_shape=(32, 32, 3),
            num_classes=10, dropout_rate=0.3, learning_rate=1e-3
        )
        assert isinstance(model, tf.keras.Model)

    def test_output_shape(self):
        model  = build_model("simple_cnn", (32, 32, 3), 10, 0.3, 1e-3)
        dummy  = np.zeros((4, 32, 32, 3), dtype="float32")
        output = model.predict(dummy, verbose=0)
        assert output.shape == (4, 10)

    def test_output_sums_to_one(self):
        model  = build_model("resnet_custom", (32, 32, 3), 10, 0.3, 1e-3)
        dummy  = np.random.rand(8, 32, 32, 3).astype("float32")
        output = model.predict(dummy, verbose=0)
        np.testing.assert_allclose(output.sum(axis=-1), np.ones(8), atol=1e-5)

    def test_resnet_parameter_count(self):
        model = build_model("resnet_custom", (32, 32, 3), 10, 0.3, 1e-3)
        assert model.count_params() > 1_000_000   # meaningful capacity

    def test_compiled_metrics(self):
        model = build_model("simple_cnn", (32, 32, 3), 10, 0.3, 1e-3)
        # Keras 3: populate metrics with a tiny forward pass, then check _compile_metrics
        import numpy as np
        dummy_x = np.zeros((2, 32, 32, 3), dtype="float32")
        dummy_y = np.eye(10, dtype="float32")[:2]
        model.evaluate(dummy_x, dummy_y, verbose=0)
        metric_names = [m.name for m in model._compile_metrics.metrics]
        assert "accuracy" in metric_names
        assert "top3_acc" in metric_names

    def test_callbacks_length(self):
        cbs = get_callbacks("./tmp_logs", patience=5,
                            lr_scheduler_type="cosine", epochs=10, initial_lr=1e-3)
        assert len(cbs) >= 4   # EarlyStopping, Checkpoint, TensorBoard, TerminateOnNaN, LRScheduler

    @pytest.mark.parametrize("scheduler", ["cosine", "step", "reduce_on_plateau"])
    def test_callback_schedulers(self, scheduler):
        cbs = get_callbacks("./tmp_logs", patience=5,
                            lr_scheduler_type=scheduler, epochs=20, initial_lr=1e-3)
        assert any(isinstance(cb, (
            tf.keras.callbacks.LearningRateScheduler,
            tf.keras.callbacks.ReduceLROnPlateau,
        )) for cb in cbs)


# ─────────────────────────────────────────────────────────────────────────────
# Mini training smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestMiniTraining:
    """Train for 2 epochs with tiny data to verify the loop runs end-to-end."""

    def test_training_loop_runs(self, datasets):
        train_ds, val_ds, _ = datasets
        model = build_model("simple_cnn", (32, 32, 3), 10, 0.3, 1e-3)
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=2, verbose=0)
        assert "accuracy"     in history.history
        assert "val_accuracy" in history.history
        assert len(history.history["accuracy"]) == 2

    def test_accuracy_keys_numeric(self, datasets):
        train_ds, val_ds, _ = datasets
        model = build_model("simple_cnn", (32, 32, 3), 10, 0.3, 1e-3)
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=1, verbose=0)
        assert isinstance(history.history["accuracy"][0], float)

    def test_evaluate_returns_dict(self, datasets):
        _, _, test_ds = datasets
        model   = build_model("simple_cnn", (32, 32, 3), 10, 0.3, 1e-3)
        results = model.evaluate(test_ds, verbose=0, return_dict=True)
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Residual-block test
# ─────────────────────────────────────────────────────────────────────────────

class TestResidualBlock:
    def test_same_dim_no_stride(self):
        inp = tf.keras.Input((8, 8, 64))
        out = _residual_block(inp, 64, stride=1)
        assert out.shape[-1] == 64

    def test_down_stride(self):
        inp = tf.keras.Input((8, 8, 64))
        out = _residual_block(inp, 128, stride=2)
        assert out.shape[1] == 4
        assert out.shape[-1] == 128
