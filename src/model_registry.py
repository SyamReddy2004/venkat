"""
model_registry.py
-----------------
Utilities for promoting, comparing and managing model versions in the
MLflow Model Registry.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)


def get_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def register_best_model(
    cfg:              dict,
    run_id:           str,
    metric_key:       str  = "test_accuracy",
    minimum_threshold: float = 0.70,
) -> Optional[str]:
    """
    Promote the model from *run_id* to the Model Registry if it meets the
    minimum accuracy threshold.

    Returns the new model version string, or None if threshold not met.
    """
    mlflow_cfg = cfg["mlflow"]
    client     = get_client(mlflow_cfg["tracking_uri"])
    model_name = mlflow_cfg["model_name"]

    # Fetch run metrics
    run_data = client.get_run(run_id).data
    metric_value = run_data.metrics.get(metric_key, 0.0)

    log.info("Run %s  %s=%.4f  (threshold=%.4f)",
             run_id, metric_key, metric_value, minimum_threshold)

    if metric_value < minimum_threshold:
        log.warning(
            "Model does NOT meet threshold (%.4f < %.4f). Skipping registration.",
            metric_value, minimum_threshold,
        )
        return None

    # Register the model
    model_uri = f"runs:/{run_id}/model"
    mv        = mlflow.register_model(model_uri, model_name)
    version   = mv.version
    log.info("Registered '%s' version %s", model_name, version)

    # Add descriptive tags
    client.set_model_version_tag(model_name, version, "metric_key",   metric_key)
    client.set_model_version_tag(model_name, version, "metric_value", f"{metric_value:.4f}")
    client.set_model_version_tag(model_name, version, "run_id",       run_id)

    # Update description
    client.update_model_version(
        name        = model_name,
        version     = version,
        description = (
            f"CIFAR-10 classifier. "
            f"{metric_key}={metric_value:.4f} on test set. "
            f"Run ID: {run_id}"
        ),
    )

    return version


def transition_to_production(
    cfg:        dict,
    version:    str,
    archive_existing: bool = True,
) -> None:
    """Move *version* to Production stage, optionally archiving older versions."""
    mlflow_cfg = cfg["mlflow"]
    client     = get_client(mlflow_cfg["tracking_uri"])
    model_name = mlflow_cfg["model_name"]

    if archive_existing:
        _archive_current_production(client, model_name)

    client.transition_model_version_stage(
        name    = model_name,
        version = version,
        stage   = "Production",
        archive_existing_versions = archive_existing,
    )
    log.info("Model '%s' version %s → Production", model_name, version)


def compare_and_promote(
    cfg:        dict,
    run_id:     str,
    metric_key: str   = "test_accuracy",
    minimum_threshold: float = 0.70,
) -> Dict[str, object]:
    """
    Register the new model, compare against current Production model, and
    promote if it is better.

    Returns a dict with registration outcome.
    """
    mlflow_cfg  = cfg["mlflow"]
    client      = get_client(mlflow_cfg["tracking_uri"])
    model_name  = mlflow_cfg["model_name"]

    # 1. Register
    new_version = register_best_model(cfg, run_id, metric_key, minimum_threshold)
    if new_version is None:
        return {"promoted": False, "reason": "below_threshold"}

    # 2. Get current production metric
    prod_metric = _get_production_metric(client, model_name, metric_key)

    # 3. Compare
    run_data     = client.get_run(run_id).data
    new_metric   = run_data.metrics.get(metric_key, 0.0)

    result = {
        "new_version":    new_version,
        "new_metric":     new_metric,
        "prod_metric":    prod_metric,
        "metric_key":     metric_key,
    }

    if prod_metric is None or new_metric > prod_metric:
        transition_to_production(cfg, new_version)
        result["promoted"] = True
        result["reason"]   = "better_than_production"
        log.info(
            "✅ New version %s promoted (%.4f > %.4f)",
            new_version, new_metric, prod_metric or 0.0,
        )
    else:
        # Still keep as Staging
        client.transition_model_version_stage(
            name    = model_name,
            version = new_version,
            stage   = "Staging",
        )
        result["promoted"] = False
        result["reason"]   = "not_better_than_production"
        log.info(
            "ℹ️  Version %s kept in Staging (%.4f ≤ %.4f)",
            new_version, new_metric, prod_metric,
        )

    return result


def list_model_versions(cfg: dict) -> None:
    """Print a table of all registered versions."""
    mlflow_cfg = cfg["mlflow"]
    client     = get_client(mlflow_cfg["tracking_uri"])
    model_name = mlflow_cfg["model_name"]

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        print("No registered models found.")
        return

    print(f"\n{'Version':<10} {'Stage':<14} {'Created':<24} {'Description'}")
    print("-" * 80)
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"  {v.version:<8} {v.current_stage:<14} "
              f"{v.creation_timestamp:<24} {v.description or '':.50s}")
    print()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _archive_current_production(client: MlflowClient, model_name: str) -> None:
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        for v in versions:
            client.transition_model_version_stage(
                name    = model_name,
                version = v.version,
                stage   = "Archived",
            )
            log.info("Archived version %s", v.version)
    except Exception as e:
        log.debug("No existing Production model to archive: %s", e)


def _get_production_metric(
    client:     MlflowClient,
    model_name: str,
    metric_key: str,
) -> Optional[float]:
    """Return the *metric_key* value of the current Production model, or None."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        return client.get_run(run_id).data.metrics.get(metric_key)
    except Exception:
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config_loader import load_config
    cfg = load_config()
    list_model_versions(cfg)
