#!/usr/bin/env python3
"""
scripts/run_evaluation.py
--------------------------
Standalone evaluation script.  Loads the latest Production (or Staging)
model from MLflow and runs comprehensive reporting.

Usage
-----
    python scripts/run_evaluation.py [--stage Production]
    python scripts/run_evaluation.py --run-id <mlflow-run-id>
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.data_preprocessing import load_cifar10, preprocess_arrays, build_tf_datasets

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 Evaluation Script")
    p.add_argument("--stage",   default="Production",
                   choices=["Production", "Staging", "None"],
                   help="MLflow model registry stage to load")
    p.add_argument("--run-id",  default=None,
                   help="Load model from a specific MLflow run (overrides --stage)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()

    import mlflow
    import mlflow.keras
    import tensorflow as tf

    from src.evaluate import evaluate

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    # ── Load model ───────────────────────────────────────────────────────
    if args.run_id:
        model_uri = f"runs:/{args.run_id}/model"
        log.info("Loading model from run: %s", model_uri)
    else:
        model_name = cfg["mlflow"]["model_name"]
        model_uri  = f"models:/{model_name}/{args.stage}"
        log.info("Loading model from registry: %s", model_uri)

    model = mlflow.keras.load_model(model_uri)

    # ── Load test data ───────────────────────────────────────────────────
    data_cfg     = cfg["data"]
    training_cfg = cfg["training"]

    x_train, y_train, x_test, y_test = load_cifar10()
    train_data, val_data, test_data   = preprocess_arrays(
        x_train, y_train, x_test, y_test,
        val_split   = data_cfg["validation_split"],
        num_classes = data_cfg["num_classes"],
        random_seed = data_cfg["random_seed"],
    )
    _, _, test_ds = build_tf_datasets(
        train_data, val_data, test_data,
        batch_size = training_cfg["batch_size"],
        augment    = False,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    import os
    output_dir = os.path.join(cfg["paths"]["logs_dir"], "evaluation", "standalone")
    metrics    = evaluate(model, test_ds, cfg, output_dir=output_dir)

    print("\nEvaluation artefacts saved to:", output_dir)
    return metrics


if __name__ == "__main__":
    main()
