"""
cifar10_training_pipeline.py
-----------------------------
Apache Airflow DAG that orchestrates the end-to-end CIFAR-10 training
pipeline with MLflow tracking.

DAG Tasks
---------
1. check_environment   – Verify dependencies and connectivity
2. preprocess_data     – Load & preprocess CIFAR-10; push summary to XCom
3. train_model         – Build model, run training with MLflow tracking
4. evaluate_model      – Full test-set evaluation; push metrics to XCom
5. register_model      – Register/promote model in MLflow Model Registry
6. generate_report     – Generate HTML & JSON performance report
7. notify              – Log pipeline completion summary

The DAG uses TaskFlow API (@task) for clean Python-native task definitions
and XCom for inter-task data sharing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Make the project root importable ─────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG default arguments
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner":            "ml-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

@dag(
    dag_id            = "cifar10_training_pipeline",
    description       = (
        "End-to-end CIFAR-10 classification pipeline: "
        "preprocessing → training → evaluation → model registration → report"
    ),
    default_args      = DEFAULT_ARGS,
    schedule_interval = "@weekly",
    start_date        = datetime(2025, 1, 1),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["ml", "cifar10", "tensorflow", "mlflow"],
    doc_md            = __doc__,
)
def cifar10_pipeline():

    # ======================================================================
    # Task 1 – Environment check
    # ======================================================================
    @task(task_id="check_environment")
    def check_environment() -> dict:
        """
        Verify that all required packages are importable and that the MLflow
        tracking server is reachable.
        """
        import importlib
        import urllib.request

        required = [
            "tensorflow", "mlflow", "sklearn",
            "numpy", "pandas", "matplotlib", "yaml",
        ]
        missing = []
        versions = {}

        for pkg in required:
            try:
                mod = importlib.import_module(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                missing.append(pkg)

        if missing:
            raise ImportError(f"Missing required packages: {missing}")

        # Check GPU availability
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        log.info("GPU devices: %s", gpus)

        # Try to reach MLflow server
        from src.config_loader import load_config
        cfg         = load_config()
        tracking_uri = cfg["mlflow"]["tracking_uri"]
        mlflow_ok    = False
        try:
            urllib.request.urlopen(tracking_uri, timeout=5)
            mlflow_ok = True
        except Exception as e:
            log.warning("MLflow server unreachable at %s: %s. "
                        "Will use local file store.", tracking_uri, e)
            # Fall back to local
            import mlflow
            mlflow.set_tracking_uri(str(_PROJECT_ROOT / "mlruns"))

        result = {
            "packages": versions,
            "missing":  missing,
            "gpu_count": len(gpus),
            "mlflow_server_reachable": mlflow_ok,
            "project_root": str(_PROJECT_ROOT),
            "status": "ok",
        }
        log.info("Environment check passed: %s", result)
        return result

    # ======================================================================
    # Task 2 – Data preprocessing
    # ======================================================================
    @task(task_id="preprocess_data")
    def preprocess_data(env_info: dict) -> dict:
        """
        Load CIFAR-10, normalise, augment configuration, compute dataset
        statistics, and return a summary dict for downstream tasks.
        """
        from src.config_loader import load_config
        from src.data_preprocessing import (
            load_cifar10, preprocess_arrays,
        )
        import numpy as np

        cfg      = load_config()
        data_cfg = cfg["data"]

        log.info("Starting data preprocessing …")
        x_train, y_train, x_test, y_test = load_cifar10()

        train_data, val_data, test_data = preprocess_arrays(
            x_train, y_train, x_test, y_test,
            val_split   = data_cfg["validation_split"],
            num_classes = data_cfg["num_classes"],
            random_seed = data_cfg["random_seed"],
        )

        x_tr, y_tr = train_data
        x_va, y_va = val_data
        x_te, y_te = test_data

        # Class distribution in training split
        y_tr_labels = np.argmax(y_tr, axis=-1)
        class_counts = {
            cfg["data"]["class_names"][i]: int(np.sum(y_tr_labels == i))
            for i in range(data_cfg["num_classes"])
        }

        summary = {
            "train_samples":  int(len(x_tr)),
            "val_samples":    int(len(x_va)),
            "test_samples":   int(len(x_te)),
            "num_classes":    data_cfg["num_classes"],
            "image_shape":    list(x_tr.shape[1:]),
            "pixel_mean":     [float(x_tr.mean(axis=(0,1,2))[c]) for c in range(3)],
            "pixel_std":      [float(x_tr.std(axis=(0,1,2))[c])  for c in range(3)],
            "class_counts":   class_counts,
            "augmentation":   data_cfg["augment"],
            "validation_split": data_cfg["validation_split"],
            "status":         "success",
        }

        log.info("Preprocessing summary:\n%s", json.dumps(summary, indent=2))
        return summary

    # ======================================================================
    # Task 3 – Model training
    # ======================================================================
    @task(task_id="train_model")
    def train_model(preprocessing_summary: dict) -> dict:
        """
        Build the model, train with MLflow tracking, and return run metadata.
        """
        import mlflow
        import tensorflow as tf

        from src.config_loader import load_config
        from src.data_preprocessing import (
            load_cifar10, preprocess_arrays, build_tf_datasets,
        )
        from src.model import build_model
        from src.train import train

        if preprocessing_summary.get("status") != "success":
            raise ValueError("Preprocessing failed; cannot train.")

        cfg          = load_config()
        data_cfg     = cfg["data"]
        training_cfg = cfg["training"]
        model_cfg    = cfg["model"]

        # ── Data ─────────────────────────────────────────────────────────
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

        # ── Model ─────────────────────────────────────────────────────────
        model = build_model(
            architecture  = model_cfg["architecture"],
            input_shape   = tuple(model_cfg["input_shape"]),
            num_classes   = model_cfg["num_classes"],
            dropout_rate  = model_cfg["dropout_rate"],
            learning_rate = training_cfg["learning_rate"],
        )

        # ── Train ─────────────────────────────────────────────────────────
        os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
        run_name = (
            f"airflow_{model_cfg['architecture']}"
            f"_ep{training_cfg['epochs']}"
            f"_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        results = train(model, train_ds, val_ds, test_ds, cfg, run_name=run_name)

        training_summary = {
            "run_id":           results.get("run_id", ""),
            "run_name":         run_name,
            "architecture":     model_cfg["architecture"],
            "epochs_requested": training_cfg["epochs"],
            "test_accuracy":    results.get("test_accuracy", 0.0),
            "test_loss":        results.get("test_loss", 0.0),
            "test_top3_acc":    results.get("test_top3_categorical_accuracy",
                                            results.get("test_top3_acc", 0.0)),
            "test_auc":         results.get("test_auc", 0.0),
            "macro_f1":         results.get("macro_f1", 0.0),
            "macro_precision":  results.get("macro_precision", 0.0),
            "macro_recall":     results.get("macro_recall", 0.0),
            "status":           "success",
        }

        log.info("Training summary:\n%s", json.dumps(training_summary, indent=2))
        return training_summary

    # ======================================================================
    # Task 4 – Evaluate
    # ======================================================================
    @task(task_id="evaluate_model")
    def evaluate_model(training_summary: dict) -> dict:
        """
        Load the freshly-trained run's model (from MLflow artefacts) and run
        a comprehensive evaluation pass.
        """
        import mlflow.keras
        import numpy as np
        import tensorflow as tf

        from src.config_loader import load_config
        from src.data_preprocessing import (
            load_cifar10, preprocess_arrays, build_tf_datasets,
        )
        from src.evaluate import evaluate

        if training_summary.get("status") != "success":
            raise ValueError("Training failed; skipping evaluation.")

        cfg           = load_config()
        data_cfg      = cfg["data"]
        training_cfg  = cfg["training"]
        run_id        = training_summary["run_id"]

        # ── Reload test data ──────────────────────────────────────────────
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

        # ── Load model from the specific run ─────────────────────────────
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        model_uri = f"runs:/{run_id}/model"
        log.info("Loading model from: %s", model_uri)
        model = mlflow.keras.load_model(model_uri)

        # ── Evaluate ─────────────────────────────────────────────────────
        output_dir = os.path.join(cfg["paths"]["logs_dir"], "evaluation", run_id)
        metrics    = evaluate(model, test_ds, cfg, output_dir=output_dir)

        # Log evaluation artefacts back to the same run
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(output_dir, artifact_path="evaluation")
            mlflow.log_metrics(
                {f"eval_{k}": v for k, v in metrics.items()
                 if isinstance(v, (int, float))}
            )

        eval_summary = {
            "run_id":      run_id,
            "output_dir":  output_dir,
            "metrics":     {k: float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float))},
            "status":      "success",
        }
        log.info("Evaluation complete. Accuracy=%.4f", metrics.get("test_accuracy", 0))
        return eval_summary

    # ======================================================================
    # Task 5 – Model registration
    # ======================================================================
    @task(task_id="register_model")
    def register_model(
        training_summary: dict,
        eval_summary:     dict,
    ) -> dict:
        """
        Compare new run with current Production model and promote if better.
        Gate: model must achieve ≥ 70% test accuracy to be registered.
        """
        from src.config_loader import load_config
        from src.model_registry import compare_and_promote, list_model_versions

        if training_summary.get("status") != "success":
            return {"promoted": False, "reason": "training_failed"}
        if eval_summary.get("status") != "success":
            return {"promoted": False, "reason": "evaluation_failed"}

        cfg    = load_config()
        run_id = training_summary["run_id"]

        # Threshold from Airflow Variable (overridable at runtime)
        try:
            threshold = float(Variable.get("cifar10_accuracy_threshold", default_var=0.70))
        except Exception:
            threshold = 0.70

        log.info("Attempting model registration for run %s (threshold=%.2f)", run_id, threshold)

        result = compare_and_promote(
            cfg                = cfg,
            run_id             = run_id,
            metric_key         = "test_accuracy",
            minimum_threshold  = threshold,
        )

        list_model_versions(cfg)   # print table to logs

        log.info("Registration result: %s", result)
        return result

    # ======================================================================
    # Task 6 – Generate HTML report
    # ======================================================================
    @task(task_id="generate_report")
    def generate_report(
        preprocessing_summary: dict,
        training_summary:      dict,
        eval_summary:          dict,
        registration_result:   dict,
    ) -> str:
        """
        Produce a self-contained HTML report and save it to disk.
        Returns the path to the report.
        """
        from src.config_loader import load_config

        cfg        = load_config()
        report_dir = os.path.join(cfg["paths"]["logs_dir"], "reports")
        os.makedirs(report_dir, exist_ok=True)

        run_id   = training_summary.get("run_id", "N/A")
        ts       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        metrics  = eval_summary.get("metrics", {})
        promoted = registration_result.get("promoted", False)

        # Metric rows for the HTML table
        metric_rows = ""
        important_metrics = [
            ("Test Accuracy",      "test_accuracy"),
            ("Top-3 Accuracy",     "test_top3_accuracy"),
            ("Macro ROC-AUC",      "test_roc_auc_macro"),
            ("Macro F1",           "macro_f1"),
            ("Weighted F1",        "weighted_f1"),
            ("Macro Precision",    "macro_precision"),
            ("Macro Recall",       "macro_recall"),
        ]
        for label, key in important_metrics:
            val = metrics.get(key, "N/A")
            fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
            pct = f"{val*100:.2f}%" if isinstance(val, float) else ""
            metric_rows += (
                f"<tr><td>{label}</td>"
                f"<td class='val'>{fmt}</td>"
                f"<td class='pct'>{pct}</td></tr>\n"
            )

        # Per-class F1 rows
        class_rows = ""
        class_names = cfg["data"]["class_names"]
        for cls in class_names:
            f1  = metrics.get(f"f1_{cls}", 0.0)
            bar = int(f1 * 100)
            class_rows += (
                f"<tr><td>{cls}</td>"
                f"<td>{f1:.4f}</td>"
                f"<td><div class='bar' style='width:{bar}%'></div></td></tr>\n"
            )

        badge_color = "#4CAF50" if promoted else "#FF9800"
        badge_text  = "✅ Promoted to Production" if promoted else "⚠️ Kept in Staging"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CIFAR-10 Pipeline Report – {ts}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f1117; color: #e0e0e0; padding: 24px;
    line-height: 1.6;
  }}
  header {{
    background: linear-gradient(135deg, #1a237e, #283593);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }}
  header h1 {{ font-size: 2rem; color: #fff; letter-spacing: -0.5px; }}
  header p  {{ color: #90caf9; margin-top: 4px; }}
  .badge {{
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    font-weight: 700; font-size: 0.85rem; margin-top: 12px;
    background: {badge_color}; color: #fff;
  }}
  section {{
    background: #1e2029; border-radius: 10px; padding: 24px;
    margin-bottom: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.3);
  }}
  h2 {{ font-size: 1.2rem; color: #90caf9; margin-bottom: 16px;
        border-bottom: 1px solid #333; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 8px 12px; background: #2a2d3a;
        color: #90caf9; font-size: 0.85rem; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #2a2d3a;
        font-size: 0.9rem; }}
  .val {{ font-family: monospace; color: #a5d6a7; font-weight: 600; }}
  .pct {{ color: #b0bec5; font-size: 0.8rem; }}
  .bar-cell {{ width: 200px; }}
  .bar {{ height: 14px; background: linear-gradient(90deg, #1565c0, #42a5f5);
          border-radius: 4px; min-width: 2px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .stat {{ background: #2a2d3a; border-radius: 8px; padding: 16px; }}
  .stat .num {{ font-size: 2rem; font-weight: 700; color: #42a5f5; }}
  .stat .lbl {{ font-size: 0.8rem; color: #78909c; margin-top: 4px; }}
  code {{ background: #2a2d3a; padding: 2px 8px; border-radius: 4px;
          font-size: 0.85rem; color: #a5d6a7; }}
  .pipeline {{ display: flex; gap: 0; flex-wrap: wrap; margin: 8px 0; }}
  .step {{ background: #1565c0; color: #fff; padding: 8px 16px;
           font-size: 0.82rem; position: relative; }}
  .step:not(:last-child)::after {{
    content: '▶'; position: absolute; right: -14px; top: 50%;
    transform: translateY(-50%); color: #42a5f5; z-index: 1;
  }}
  .step:not(:first-child) {{ margin-left: 16px; }}
  @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<header>
  <h1>🧠 CIFAR-10 MLOps Pipeline Report</h1>
  <p>Generated: <strong>{ts}</strong> &nbsp;|&nbsp; Run ID: <code>{run_id}</code></p>
  <p>Architecture: <code>{training_summary.get('architecture','N/A')}</code></p>
  <div class="badge">{badge_text}</div>
</header>

<!-- Pipeline Flow -->
<section>
  <h2>🔄 Pipeline DAG</h2>
  <div class="pipeline">
    <div class="step">1. check_environment</div>
    <div class="step">2. preprocess_data</div>
    <div class="step">3. train_model</div>
    <div class="step">4. evaluate_model</div>
    <div class="step">5. register_model</div>
    <div class="step">6. generate_report</div>
    <div class="step">7. notify</div>
  </div>
</section>

<!-- Dataset Summary -->
<section>
  <h2>📦 Dataset Summary</h2>
  <div class="grid">
    <div class="stat"><div class="num">{preprocessing_summary.get('train_samples','—'):,}</div>
      <div class="lbl">Training samples</div></div>
    <div class="stat"><div class="num">{preprocessing_summary.get('val_samples','—'):,}</div>
      <div class="lbl">Validation samples</div></div>
    <div class="stat"><div class="num">{preprocessing_summary.get('test_samples','—'):,}</div>
      <div class="lbl">Test samples</div></div>
    <div class="stat"><div class="num">{preprocessing_summary.get('num_classes','—')}</div>
      <div class="lbl">Classes</div></div>
  </div>
</section>

<!-- Key Metrics -->
<section>
  <h2>📊 Performance Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th><th>Percentage</th></tr></thead>
    <tbody>
    {metric_rows}
    </tbody>
  </table>
</section>

<!-- Per-class F1 -->
<section>
  <h2>🎯 Per-Class F1 Score</h2>
  <table>
    <thead><tr><th>Class</th><th>F1 Score</th><th style="width:220px">Bar</th></tr></thead>
    <tbody>
    {class_rows}
    </tbody>
  </table>
</section>

<!-- Training Hyperparameters -->
<section>
  <h2>⚙️ Hyperparameters</h2>
  <table>
    <tbody>
    <tr><td>Architecture</td><td class="val">{training_summary.get('architecture','N/A')}</td></tr>
    <tr><td>Epochs (requested)</td><td class="val">{training_summary.get('epochs_requested','N/A')}</td></tr>
    <tr><td>Dataset</td><td class="val">CIFAR-10</td></tr>
    <tr><td>MLflow Experiment</td><td class="val">{cfg['mlflow']['experiment_name']}</td></tr>
    <tr><td>Registered Model</td><td class="val">{cfg['mlflow']['model_name']}</td></tr>
    </tbody>
  </table>
</section>

<!-- Registration -->
<section>
  <h2>🏷️ Model Registration</h2>
  <table>
    <tbody>
    <tr><td>New Version</td>
        <td class="val">{registration_result.get('new_version','N/A')}</td></tr>
    <tr><td>New Accuracy</td>
        <td class="val">{registration_result.get('new_metric', 0.0):.4f}</td></tr>
    <tr><td>Previous Production Accuracy</td>
        <td class="val">{registration_result.get('prod_metric') or 'N/A'}</td></tr>
    <tr><td>Promoted</td>
        <td class="val" style="color:{'#4CAF50' if promoted else '#FF9800'}">
        {'Yes' if promoted else 'No'}</td></tr>
    <tr><td>Reason</td>
        <td class="val">{registration_result.get('reason','N/A')}</td></tr>
    </tbody>
  </table>
</section>

</body>
</html>"""

        report_path = os.path.join(report_dir, f"report_{run_id}.html")
        with open(report_path, "w") as f:
            f.write(html)

        # Also save JSON summary
        json_path = os.path.join(report_dir, f"report_{run_id}.json")
        full_report = {
            "run_id":              run_id,
            "timestamp":           ts,
            "preprocessing":       preprocessing_summary,
            "training":            training_summary,
            "evaluation_metrics":  eval_summary.get("metrics", {}),
            "registration":        registration_result,
        }
        with open(json_path, "w") as f:
            json.dump(full_report, f, indent=2, default=str)

        # Log report artefact to MLflow
        try:
            import mlflow
            from src.config_loader import load_config as _lc
            _cfg = _lc()
            mlflow.set_tracking_uri(_cfg["mlflow"]["tracking_uri"])
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(report_path, artifact_path="reports")
                mlflow.log_artifact(json_path,   artifact_path="reports")
        except Exception as e:
            log.warning("Could not log report to MLflow: %s", e)

        log.info("Report saved to: %s", report_path)
        return report_path

    # ======================================================================
    # Task 7 – Notify
    # ======================================================================
    @task(task_id="notify")
    def notify(
        training_summary:    dict,
        eval_summary:        dict,
        registration_result: dict,
        report_path:         str,
    ) -> None:
        """
        Log a rich pipeline completion summary to Airflow task logs.
        (Extend to send email / Slack notifications as needed.)
        """
        sep = "=" * 72
        acc = training_summary.get("test_accuracy", 0.0)
        f1  = training_summary.get("macro_f1",      eval_summary.get("metrics", {}).get("macro_f1", 0.0))

        print(f"\n{sep}")
        print("  CIFAR-10 PIPELINE COMPLETE")
        print(sep)
        print(f"  Run ID        : {training_summary.get('run_id','N/A')}")
        print(f"  Architecture  : {training_summary.get('architecture','N/A')}")
        print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Macro F1      : {f1:.4f}")
        print(f"  Promoted      : {'✅ Yes' if registration_result.get('promoted') else '❌ No'}")
        print(f"  Report        : {report_path}")
        print(sep + "\n")

    # ======================================================================
    # Wire up the DAG
    # ======================================================================
    env_info               = check_environment()
    preproc_summary        = preprocess_data(env_info)
    training_summary       = train_model(preproc_summary)
    eval_summary           = evaluate_model(training_summary)
    registration_result    = register_model(training_summary, eval_summary)
    report_path            = generate_report(
                                 preproc_summary,
                                 training_summary,
                                 eval_summary,
                                 registration_result,
                             )
    notify(training_summary, eval_summary, registration_result, report_path)


# Instantiate the DAG
cifar10_dag = cifar10_pipeline()
