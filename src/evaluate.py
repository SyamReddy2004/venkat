"""
evaluate.py
-----------
Standalone evaluation module.  Loads the latest Production-stage model from
MLflow Model Registry and runs a comprehensive evaluation on the CIFAR-10
test set, producing a rich performance report.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import tensorflow as tf

log = logging.getLogger(__name__)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    model:    tf.keras.Model | None,
    test_ds:  tf.data.Dataset,
    cfg:      dict,
    stage:    str = "Production",
    output_dir: str | None = None,
) -> Dict[str, float]:
    """
    Run full evaluation and return a dict of metrics.

    If *model* is None the function loads the model registered in MLflow
    at *stage* from the model registry.
    """
    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    if model is None:
        model = _load_registered_model(mlflow_cfg["model_name"], stage)

    output_dir = output_dir or os.path.join(cfg["paths"]["logs_dir"], "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect predictions
    # ------------------------------------------------------------------
    y_true, y_pred_prob = _collect_predictions(model, test_ds)
    y_pred              = np.argmax(y_pred_prob, axis=-1)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------
    metrics: Dict[str, float] = {}

    # Accuracy
    metrics["test_accuracy"] = float(np.mean(y_true == y_pred))

    # Top-3 accuracy
    top3 = np.argsort(y_pred_prob, axis=-1)[:, -3:]
    metrics["test_top3_accuracy"] = float(
        np.mean([y_true[i] in top3[i] for i in range(len(y_true))])
    )

    # Macro ROC-AUC (one-vs-rest)
    n_classes = y_pred_prob.shape[-1]
    y_true_bin = np.eye(n_classes)[y_true]
    metrics["test_roc_auc_macro"] = float(
        roc_auc_score(y_true_bin, y_pred_prob, multi_class="ovr", average="macro")
    )

    # Classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    report_str  = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    metrics["macro_f1"]       = report_dict["macro avg"]["f1-score"]
    metrics["weighted_f1"]    = report_dict["weighted avg"]["f1-score"]
    metrics["macro_precision"] = report_dict["macro avg"]["precision"]
    metrics["macro_recall"]    = report_dict["macro avg"]["recall"]

    for cls in CLASS_NAMES:
        metrics[f"f1_{cls}"]        = report_dict[cls]["f1-score"]
        metrics[f"precision_{cls}"] = report_dict[cls]["precision"]
        metrics[f"recall_{cls}"]    = report_dict[cls]["recall"]

    log.info("Evaluation complete.\n%s", report_str)

    # ------------------------------------------------------------------
    # Save detailed report
    # ------------------------------------------------------------------
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    df = pd.DataFrame(report_dict).T
    df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_confusion_matrix(y_true, y_pred, output_dir)
    _plot_roc_curves(y_true_bin, y_pred_prob, output_dir)
    _plot_precision_recall(y_true_bin, y_pred_prob, output_dir)
    _plot_confidence_histogram(y_pred_prob, y_true, output_dir)

    # ------------------------------------------------------------------
    # Print human-readable summary
    # ------------------------------------------------------------------
    _print_summary_table(metrics, report_dict)

    return metrics


def load_and_evaluate(cfg: dict, stage: str = "Production") -> Dict[str, float]:
    """Load Production model + CIFAR-10 test set and run evaluate()."""
    from src.data_preprocessing import (
        load_cifar10, preprocess_arrays, build_tf_datasets,
    )
    data_cfg     = cfg["data"]
    training_cfg = cfg["training"]

    x_train, y_train, x_test, y_test = load_cifar10()
    _, _, test_data = preprocess_arrays(
        x_train, y_train, x_test, y_test,
        val_split   = data_cfg["validation_split"],
        num_classes = data_cfg["num_classes"],
        random_seed = data_cfg["random_seed"],
    )
    dummy_train = (x_train[:1] / 255.0, np.zeros((1, data_cfg["num_classes"])))
    dummy_val   = dummy_train

    _, _, test_ds = build_tf_datasets(
        dummy_train, dummy_val, test_data,
        batch_size = training_cfg["batch_size"],
        augment    = False,
    )
    return evaluate(None, test_ds, cfg, stage=stage)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_registered_model(model_name: str, stage: str) -> tf.keras.Model:
    log.info("Loading registered model '%s' @ stage='%s'", model_name, stage)
    model_uri = f"models:/{model_name}/{stage}"
    model     = mlflow.keras.load_model(model_uri)
    log.info("Model loaded: %s", model_uri)
    return model


def _collect_predictions(
    model:   tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_list, y_pred_list = [], []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred_list.append(preds)
        if y_batch.shape.rank > 1 and y_batch.shape[-1] > 1:
            y_true_list.append(np.argmax(y_batch.numpy(), axis=-1))
        else:
            y_true_list.append(y_batch.numpy().flatten())
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def _plot_confusion_matrix(y_true, y_pred, out_dir):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(10), yticks=range(10),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True",
           title="CIFAR-10 – Normalised Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thr = cm_norm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > thr else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _plot_roc_curves(y_true_bin, y_pred_prob, out_dir):
    n = y_pred_prob.shape[1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        auc_i       = roc_auc_score(y_true_bin[:, i], y_pred_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=1.5, label=f"{cls} (AUC={auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves – One-vs-Rest")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)


def _plot_precision_recall(y_true_bin, y_pred_prob, out_dir):
    n    = y_pred_prob.shape[1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        ap           = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])
        ax.plot(rec, prec, color=col, lw=1.5, label=f"{cls} (AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves – One-vs-Rest")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "precision_recall_curves.png"), dpi=150)
    plt.close(fig)


def _plot_confidence_histogram(y_pred_prob, y_true, out_dir):
    correct_conf   = y_pred_prob[np.arange(len(y_true)), y_true]
    predicted_conf = y_pred_prob.max(axis=-1)
    wrong_mask     = np.argmax(y_pred_prob, axis=-1) != y_true

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(predicted_conf[~wrong_mask], bins=50, alpha=0.7,
            color="#4CAF50", label="Correct predictions")
    ax.hist(predicted_conf[wrong_mask],  bins=50, alpha=0.7,
            color="#f44336", label="Wrong predictions")
    ax.set(xlabel="Predicted Confidence", ylabel="Count",
           title="Confidence Distribution – Correct vs Wrong")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_histogram.png"), dpi=150)
    plt.close(fig)


def _print_summary_table(metrics: dict, report_dict: dict) -> None:
    sep   = "=" * 70
    print(f"\n{sep}")
    print("  CIFAR-10 EVALUATION SUMMARY")
    print(sep)
    print(f"  {'Metric':<30} {'Value':>10}")
    print("-" * 70)
    key_metrics = [
        ("Test Accuracy",          "test_accuracy"),
        ("Top-3 Accuracy",         "test_top3_accuracy"),
        ("Macro ROC-AUC",          "test_roc_auc_macro"),
        ("Macro F1",               "macro_f1"),
        ("Weighted F1",            "weighted_f1"),
        ("Macro Precision",        "macro_precision"),
        ("Macro Recall",           "macro_recall"),
    ]
    for label, key in key_metrics:
        if key in metrics:
            print(f"  {label:<30} {metrics[key]:>10.4f}")
    print(sep)
    print("\n  Per-class F1 Scores:")
    print("-" * 40)
    for cls in CLASS_NAMES:
        f1 = report_dict[cls]["f1-score"]
        bar = "█" * int(f1 * 20)
        print(f"  {cls:<12} {f1:.3f}  {bar}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config_loader import load_config
    cfg = load_config()
    load_and_evaluate(cfg)
