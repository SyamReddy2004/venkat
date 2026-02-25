"""
model.py
--------
Defines multiple model architectures for CIFAR-10 classification.

Available architectures
-----------------------
- ``simple_cnn``         – Lightweight baseline (test quickly)
- ``resnet_custom``      – Custom ResNet-style model with residual blocks
- ``efficientnet_transfer`` – Transfer learning via EfficientNetB0
"""
from __future__ import annotations

import logging
from typing import Literal

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

log = logging.getLogger(__name__)

Architecture = Literal["simple_cnn", "resnet_custom", "efficientnet_transfer"]

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def build_model(
    architecture: Architecture = "resnet_custom",
    input_shape:  tuple        = (32, 32, 3),
    num_classes:  int          = 10,
    dropout_rate: float        = 0.4,
    learning_rate: float       = 0.001,
) -> tf.keras.Model:
    """Build, compile, and return the requested model."""

    builders = {
        "simple_cnn":            _build_simple_cnn,
        "resnet_custom":         _build_resnet_custom,
        "efficientnet_transfer": _build_efficientnet_transfer,
    }

    if architecture not in builders:
        raise ValueError(f"Unknown architecture: {architecture!r}. "
                         f"Choose from {list(builders)}")

    log.info("Building model: %s", architecture)
    model = builders[architecture](input_shape, num_classes, dropout_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer = optimizer,
        loss      = "categorical_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    log.info("Model built. Parameters: {:,}".format(model.count_params()))
    return model


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

def _build_simple_cnn(
    input_shape: tuple,
    num_classes: int,
    dropout_rate: float,
) -> tf.keras.Model:
    """3-block VGG-style CNN – fast baseline."""
    inputs = layers.Input(shape=input_shape)

    x = _conv_bn_relu(inputs, 32, 3)
    x = _conv_bn_relu(x,      32, 3)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = _conv_bn_relu(x, 64, 3)
    x = _conv_bn_relu(x, 64, 3)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = _conv_bn_relu(x, 128, 3)
    x = _conv_bn_relu(x, 128, 3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="simple_cnn")


def _build_resnet_custom(
    input_shape:  tuple,
    num_classes:  int,
    dropout_rate: float,
) -> tf.keras.Model:
    """Custom ResNet with 4 residual stages tailored for 32×32 inputs."""
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Stage 1
    x = _residual_block(x, 64, stride=1)
    x = _residual_block(x, 64, stride=1)

    # Stage 2
    x = _residual_block(x, 128, stride=2)
    x = _residual_block(x, 128, stride=1)

    # Stage 3
    x = _residual_block(x, 256, stride=2)
    x = _residual_block(x, 256, stride=1)
    x = _residual_block(x, 256, stride=1)

    # Stage 4
    x = _residual_block(x, 512, stride=2)
    x = _residual_block(x, 512, stride=1)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="resnet_custom")


def _build_efficientnet_transfer(
    input_shape:  tuple,
    num_classes:  int,
    dropout_rate: float,
) -> tf.keras.Model:
    """EfficientNetB0 with frozen base + custom head (transfer learning)."""
    # Resize layer since EfficientNet expects ≥ 32×32 but benefits from larger
    inputs  = layers.Input(shape=input_shape)
    resized = layers.Resizing(96, 96)(inputs)          # upsample for better features
    scaled  = layers.Rescaling(1.0 / 255.0)(resized)   # EfficientNet wants [0,1] if not using include_preprocessing

    base = tf.keras.applications.EfficientNetB0(
        include_top   = False,
        weights       = "imagenet",
        input_tensor  = scaled,
    )
    base.trainable = False   # freeze during Phase-1

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="efficientnet_transfer")


# ---------------------------------------------------------------------------
# Residual block helpers
# ---------------------------------------------------------------------------

def _residual_block(
    x:       tf.Tensor,
    filters: int,
    stride:  int = 1,
) -> tf.Tensor:
    """Pre-activation residual block (He et al., 2016)."""
    shortcut = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False
        )(x)

    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)

    return layers.Add()([x, shortcut])


def _conv_bn_relu(x: tf.Tensor, filters: int, kernel: int) -> tf.Tensor:
    x = layers.Conv2D(filters, kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------

def get_callbacks(
    log_dir:            str,
    patience:           int   = 7,
    lr_scheduler_type:  str   = "cosine",
    epochs:             int   = 30,
    initial_lr:         float = 0.001,
) -> list:
    """Return a standard set of Keras callbacks."""
    import os, math

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = patience,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath        = os.path.join(log_dir, "best_model.keras"),
            monitor         = "val_accuracy",
            save_best_only  = True,
            save_weights_only = False,
            verbose         = 1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir        = log_dir,
            histogram_freq = 1,
            update_freq    = "epoch",
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    # Learning-rate schedule
    if lr_scheduler_type == "cosine":
        def cosine_schedule(epoch: int, _lr: float) -> float:
            return initial_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(cosine_schedule, verbose=0)
        )
    elif lr_scheduler_type == "step":
        def step_schedule(epoch: int, lr: float) -> float:
            return lr * 0.5 if epoch % 10 == 0 and epoch > 0 else lr
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(step_schedule, verbose=0)
        )
    else:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            )
        )

    return callbacks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = build_model("resnet_custom")
    m.summary()
