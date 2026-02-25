"""
train.py
--------
Training loop with full MLflow experiment tracking.

Tracked artefacts
-----------------
- Hyperparameters (via mlflow.log_params)
- Per-epoch metrics (via mlflow.log_metrics)
- Final metrics: accuracy, top-3 accuracy, AUC, loss
- Keras model (via mlflow.keras.log_model)
- TensorBoard logs
- Per-class metrics CSV
- Confusion-matrix image (PNG)
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

log = logging.getLogger(__name__)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# MLflow callback – logs metrics after every epoch
# ---------------------------------------------------------------------------

class MLflowCallback(tf.keras.callbacks.Callback):
    """Streams Keras epoch metrics to the active MLflow run."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs:
            mlflow.log_metrics(
                {k: float(v) for k, v in logs.items()},
                step=epoch,
            )


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    model:     tf.keras.Model,
    train_ds:  tf.data.Dataset,
    val_ds:    tf.data.Dataset,
    test_ds:   tf.data.Dataset,
    cfg:       dict,
    run_name:  str | None = None,
) -> Dict[str, float]:
    """
    Run a full training experiment tracked by MLflow.

    Returns
    -------
    dict of final test metrics.
    """
    from src.model import get_callbacks

    training_cfg = cfg["training"]
    mlflow_cfg   = cfg["mlflow"]
    model_cfg    = cfg["model"]
    paths_cfg    = cfg["paths"]

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    run_name = run_name or (
        f"{model_cfg['architecture']}"
        f"_bs{training_cfg['batch_size']}"
        f"_lr{training_cfg['learning_rate']}"
        f"_ep{training_cfg['epochs']}"
        f"_{int(time.time())}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        log.info("MLflow run: %s  (id=%s)", run_name, run.info.run_id)

        # ---------------------------------------------------------------
        # Log hyperparameters
        # ---------------------------------------------------------------
        params = {
            "architecture":         model_cfg["architecture"],
            "epochs":               training_cfg["epochs"],
            "batch_size":           training_cfg["batch_size"],
            "learning_rate":        training_cfg["learning_rate"],
            "lr_scheduler":         training_cfg["lr_scheduler"],
            "dropout_rate":         model_cfg["dropout_rate"],
            "optimizer":            training_cfg["optimizer"],
            "early_stopping_patience": training_cfg["early_stopping_patience"],
            "num_classes":          model_cfg["num_classes"],
            "input_shape":          str(model_cfg["input_shape"]),
            "trainable_params":     model.count_params(),
            "dataset":              cfg["data"]["dataset"],
            "augmentation":         cfg["data"]["augment"],
        }
        mlflow.log_params(params)
        log.info("Logged %d hyperparameters", len(params))

        # ---------------------------------------------------------------
        # Log model summary as text artefact
        # ---------------------------------------------------------------
        summary_buf = io.StringIO()
        model.summary(print_fn=lambda line: summary_buf.write(line + "\n"))
        mlflow.log_text(summary_buf.getvalue(), "model_summary.txt")

        # ---------------------------------------------------------------
        # Build callbacks
        # ---------------------------------------------------------------
        log_dir = os.path.join(paths_cfg["logs_dir"], run_name)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = get_callbacks(
            log_dir           = log_dir,
            patience          = training_cfg["early_stopping_patience"],
            lr_scheduler_type = training_cfg["lr_scheduler"],
            epochs            = training_cfg["epochs"],
            initial_lr        = training_cfg["learning_rate"],
        )
        callbacks.append(MLflowCallback())

        # ---------------------------------------------------------------
        # Train
        # ---------------------------------------------------------------
        log.info("Starting training …")
        t0 = time.time()
        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = training_cfg["epochs"],
            callbacks       = callbacks,
            verbose         = 1,
        )
        training_time = time.time() - t0
        mlflow.log_metric("training_time_seconds", training_time)
        log.info("Training finished in %.1f s", training_time)

        # ---------------------------------------------------------------
        # Evaluate on test set
        # ---------------------------------------------------------------
        log.info("Evaluating on test set …")
        test_results = model.evaluate(test_ds, verbose=0, return_dict=True)
        test_metrics = {f"test_{k}": float(v) for k, v in test_results.items()}
        mlflow.log_metrics(test_metrics)
        log.info("Test metrics: %s", test_metrics)

        # ---------------------------------------------------------------
        # Detailed evaluation (classification report + confusion matrix)
        # ---------------------------------------------------------------
        y_true, y_pred_prob = _collect_predictions(model, test_ds)
        report_metrics = _log_detailed_metrics(
            y_true, y_pred_prob, run, log_dir
        )
        test_metrics.update(report_metrics)

        # ---------------------------------------------------------------
        # Log training history plots
        # ---------------------------------------------------------------
        _log_history_plots(history, log_dir)

        # ---------------------------------------------------------------
        # Log the Keras model to MLflow Model Registry
        # ---------------------------------------------------------------
        log.info("Logging model to MLflow …")
        input_example = np.zeros((1, *model_cfg["input_shape"]), dtype="float32")
        mlflow.keras.log_model(
            model,
            artifact_path   = "model",
            input_example   = input_example,
            registered_model_name = mlflow_cfg["model_name"],
        )

        # ---------------------------------------------------------------
        # Log TensorBoard directory
        # ---------------------------------------------------------------
        mlflow.log_artifacts(log_dir, artifact_path="tensorboard")

        log.info("MLflow run complete: %s", run.info.run_id)
        test_metrics["run_id"] = run.info.run_id
        return test_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_predictions(
    model:   tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect ground-truth labels and predicted probabilities."""
    y_true_list, y_pred_list = [], []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred_list.append(preds)
        # y_batch may be one-hot encoded
        if y_batch.shape.rank > 1 and y_batch.shape[-1] > 1:
            y_true_list.append(np.argmax(y_batch.numpy(), axis=-1))
        else:
            y_true_list.append(y_batch.numpy().flatten())
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def _log_detailed_metrics(
    y_true:      np.ndarray,
    y_pred_prob: np.ndarray,
    run:         mlflow.ActiveRun,
    log_dir:     str,
) -> Dict[str, float]:
    """Log per-class report, confusion matrix, and macro ROC-AUC."""
    y_pred = np.argmax(y_pred_prob, axis=-1)

    # ---- Classification report ----------------------------------------
    report = classification_report(
        y_true, y_pred,
        target_names = CLASS_NAMES,
        output_dict  = True,
    )
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    mlflow.log_text(report_str, "classification_report.txt")
    log.info("Classification Report:\n%s", report_str)

    # Save per-class F1 as individual metrics
    extra_metrics: Dict[str, float] = {}
    for cls in CLASS_NAMES:
        if cls in report:
            extra_metrics[f"f1_{cls}"] = report[cls]["f1-score"]
    extra_metrics["macro_f1"]       = report["macro avg"]["f1-score"]
    extra_metrics["weighted_f1"]    = report["weighted avg"]["f1-score"]
    extra_metrics["macro_precision"] = report["macro avg"]["precision"]
    extra_metrics["macro_recall"]    = report["macro avg"]["recall"]
    mlflow.log_metrics(extra_metrics)

    # ---- ROC-AUC (one-vs-rest, macro) ---------------------------------
    try:
        n_classes = y_pred_prob.shape[-1]
        y_true_bin = np.eye(n_classes)[y_true]
        auc_macro = roc_auc_score(y_true_bin, y_pred_prob, multi_class="ovr", average="macro")
        mlflow.log_metric("test_roc_auc_macro", float(auc_macro))
        extra_metrics["test_roc_auc_macro"] = float(auc_macro)
    except Exception as e:
        log.warning("Could not compute ROC-AUC: %s", e)

    # ---- Confusion matrix image ----------------------------------------
    cm_path = _plot_confusion_matrix(y_true, y_pred, log_dir)
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # ---- Per-class accuracy chart ------------------------------------
    acc_path = _plot_per_class_accuracy(report, log_dir)
    mlflow.log_artifact(acc_path, artifact_path="plots")

    return extra_metrics


def _plot_confusion_matrix(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    log_dir:  str,
) -> str:
    """Save confusion-matrix as PNG; return file path."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks      = range(10),
        yticks      = range(10),
        xticklabels = CLASS_NAMES,
        yticklabels = CLASS_NAMES,
        ylabel      = "True label",
        xlabel      = "Predicted label",
        title       = "Normalised Confusion Matrix – CIFAR-10",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i,
                f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                ha="center", va="center", fontsize=7,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    path = os.path.join(log_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_per_class_accuracy(report: dict, log_dir: str) -> str:
    """Bar chart of per-class precision, recall, F1."""
    metrics   = ["precision", "recall", "f1-score"]
    n_classes = len(CLASS_NAMES)
    x         = np.arange(n_classes)
    width     = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#4CAF50", "#2196F3", "#FF5722"]

    for i, metric in enumerate(metrics):
        vals = [report[cls][metric] for cls in CLASS_NAMES]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=colors[i], alpha=0.85)

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics – CIFAR-10")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(log_dir, "per_class_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _log_history_plots(history: tf.keras.callbacks.History, log_dir: str) -> None:
    """Save accuracy and loss curves as PNGs and log to MLflow."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="#FF5722")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", color="#2196F3")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#FF5722")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(log_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    from src.config_loader import load_config
    from src.data_preprocessing import (
        load_cifar10, preprocess_arrays, build_tf_datasets,
    )
    from src.model import build_model

    cfg = load_config()
    data_cfg     = cfg["data"]
    training_cfg = cfg["training"]
    model_cfg    = cfg["model"]

    # Data
    x_train, y_train, x_test, y_test = load_cifar10()
    train_data, val_data, test_data   = preprocess_arrays(
        x_train, y_train, x_test, y_test,
        val_split   = data_cfg["validation_split"],
        num_classes = data_cfg["num_classes"],
        random_seed = data_cfg["random_seed"],
    )
    train_ds, val_ds, test_ds = build_tf_datasets(
        train_data, val_data, test_data,
        batch_size = training_cfg["batch_size"],
        augment    = data_cfg["augment"],
        seed       = data_cfg["random_seed"],
    )

    # Model
    model = build_model(
        architecture  = model_cfg["architecture"],
        input_shape   = tuple(model_cfg["input_shape"]),
        num_classes   = model_cfg["num_classes"],
        dropout_rate  = model_cfg["dropout_rate"],
        learning_rate = training_cfg["learning_rate"],
    )

    # Train
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    results = train(model, train_ds, val_ds, test_ds, cfg)
    print("\nFinal test metrics:")
    for k, v in results.items():
        print(f"  {k}: {v}")
