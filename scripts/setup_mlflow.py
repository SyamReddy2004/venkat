#!/usr/bin/env python3
"""
scripts/setup_mlflow.py
------------------------
Initialize the MLflow experiment and model registry.
Run this ONCE before starting training.

Usage
-----
    python scripts/setup_mlflow.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def main():
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.config_loader import load_config

    cfg        = load_config()
    mlflow_cfg = cfg["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = MlflowClient(tracking_uri=mlflow_cfg["tracking_uri"])

    # ── Create / get experiment ──────────────────────────────────────────
    exp_name = mlflow_cfg["experiment_name"]
    exp      = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            exp_name,
            artifact_location = mlflow_cfg.get("artifact_location", "./mlruns"),
        )
        log.info("Created experiment '%s'  (id=%s)", exp_name, exp_id)
    else:
        log.info("Experiment '%s' already exists  (id=%s)", exp_name, exp.experiment_id)

    # ── Create / get registered model ────────────────────────────────────
    model_name = mlflow_cfg["model_name"]
    try:
        client.create_registered_model(
            model_name,
            description = (
                "CIFAR-10 multiclass image classifier trained with TensorFlow. "
                "Tracks accuracy, F1, ROC-AUC across all 10 classes."
            ),
        )
        client.set_registered_model_tag(model_name, "framework",  "TensorFlow")
        client.set_registered_model_tag(model_name, "dataset",    "CIFAR-10")
        client.set_registered_model_tag(model_name, "task",       "image_classification")
        log.info("Registered model '%s' created.", model_name)
    except Exception as e:
        log.info("Registered model '%s' already exists: %s", model_name, e)

    print("\n✅ MLflow setup complete.")
    print(f"   Tracking URI : {mlflow_cfg['tracking_uri']}")
    print(f"   Experiment   : {exp_name}")
    print(f"   Model name   : {model_name}")
    print(f"\n   Start the UI with:  mlflow ui --port 5000")


if __name__ == "__main__":
    main()
