"""
data_preprocessing.py
----------------------
Handles loading, normalisation, augmentation, and splitting of the CIFAR-10
dataset.  All functions are usable standalone or called by the Airflow DAG.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (or use cache) and return raw CIFAR-10 arrays.

    Returns
    -------
    (x_train, y_train, x_test, y_test) – uint8 arrays, labels are 1-D.
    """
    log.info("Loading CIFAR-10 dataset …")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    log.info("Train: %s  Test: %s", x_train.shape, x_test.shape)
    return x_train, y_train, x_test, y_test


def preprocess_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test:  np.ndarray,
    y_test:  np.ndarray,
    val_split:    float = 0.1,
    num_classes:  int   = 10,
    random_seed:  int   = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Normalise, convert labels, and carve out a validation split.

    Returns
    -------
    (train_ds, val_ds, test_ds)  where each is (x, y_onehot).
    """
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Normalise pixel values to [0, 1]
    # ------------------------------------------------------------------
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ------------------------------------------------------------------
    # Per-channel mean / std normalisation (computed on train only)
    # ------------------------------------------------------------------
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std  = x_train.std(axis=(0, 1, 2),  keepdims=True) + 1e-7
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    # ------------------------------------------------------------------
    # Validation split
    # ------------------------------------------------------------------
    n_val   = int(len(x_train) * val_split)
    indices = rng.permutation(len(x_train))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    x_val, y_val     = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    # ------------------------------------------------------------------
    # One-hot encode labels
    # ------------------------------------------------------------------
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes)

    log.info(
        "Splits → train: %d  val: %d  test: %d",
        len(x_train), len(x_val), len(x_test),
    )
    return (x_train, y_train_oh), (x_val, y_val_oh), (x_test, y_test_oh)


def build_tf_datasets(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data:   Tuple[np.ndarray, np.ndarray],
    test_data:  Tuple[np.ndarray, np.ndarray],
    batch_size: int  = 64,
    augment:    bool = True,
    seed:       int  = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Wrap numpy arrays in optimised tf.data pipelines.

    The training pipeline applies random augmentation; val/test are plain.
    """
    x_train, y_train = train_data
    x_val,   y_val   = val_data
    x_test,  y_test  = test_data

    AUTOTUNE = tf.data.AUTOTUNE

    # --- Train -------------------------------------------------------
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=seed)
        .batch(batch_size)
        .map(
            lambda x, y: (
                tf.map_fn(_augment, x, fn_output_signature=tf.float32)
                if augment else x,
                y,
            ),
            num_parallel_calls=AUTOTUNE,
        )
        .prefetch(AUTOTUNE)
    )

    # --- Validation --------------------------------------------------
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # --- Test ---------------------------------------------------------
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _augment(image: tf.Tensor) -> tf.Tensor:
    """Random crop, horizontal flip, and colour jitter.

    Operates on a *single* image tensor of shape (H, W, C).
    Called via tf.map_fn so that the batch dimension is never present here.
    """
    # Pad then random-crop back to original size
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)  # → (36, 36, C)
    image = tf.image.random_crop(image, size=tf.constant([32, 32, 3]))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    return image


# ---------------------------------------------------------------------------
# Convenience wrapper used by the Airflow task
# ---------------------------------------------------------------------------

def run_preprocessing(cfg: dict) -> dict:
    """End-to-end preprocessing; returns a summary dict for XCom."""
    data_cfg     = cfg["data"]
    training_cfg = cfg["training"]

    x_train, y_train, x_test, y_test = load_cifar10()
    train_data, val_data, test_data  = preprocess_arrays(
        x_train, y_train, x_test, y_test,
        val_split   = data_cfg["validation_split"],
        num_classes = data_cfg["num_classes"],
        random_seed = data_cfg["random_seed"],
    )

    summary = {
        "train_samples": int(len(train_data[0])),
        "val_samples":   int(len(val_data[0])),
        "test_samples":  int(len(test_data[0])),
        "num_classes":   data_cfg["num_classes"],
        "image_shape":   list(train_data[0].shape[1:]),
        "status":        "success",
    }
    log.info("Preprocessing summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config_loader import load_config
    cfg = load_config()
    result = run_preprocessing(cfg)
    print("Preprocessing result:", result)
