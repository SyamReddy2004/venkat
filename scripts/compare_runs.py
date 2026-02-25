#!/usr/bin/env python3
"""
scripts/compare_runs.py
------------------------
Compare two or more MLflow runs side-by-side and print a ranked table.

Usage
-----
    python scripts/compare_runs.py                      # compare all runs
    python scripts/compare_runs.py --top 5             # top 5 by accuracy
    python scripts/compare_runs.py --metric macro_f1   # rank by macro F1
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top",    type=int, default=10,         help="Number of top runs to show")
    p.add_argument("--metric", default="test_accuracy",      help="Primary sort metric")
    return p.parse_args()


def main():
    args = parse_args()

    import mlflow
    import pandas as pd
    from src.config_loader import load_config

    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    if experiment is None:
        print("No experiment found. Run a training job first.")
        return

    runs = mlflow.search_runs(
        experiment_ids = [experiment.experiment_id],
        order_by       = [f"metrics.{args.metric} DESC"],
        max_results    = args.top,
    )

    if runs.empty:
        print("No runs found.")
        return

    # Select display columns
    display_cols = ["run_id", "tags.mlflow.runName"]
    metric_cols  = [
        f"metrics.{args.metric}",
        "metrics.test_accuracy",
        "metrics.test_top3_categorical_accuracy",
        "metrics.macro_f1",
        "metrics.test_roc_auc_macro",
    ]
    param_cols   = [
        "params.architecture",
        "params.epochs",
        "params.batch_size",
        "params.learning_rate",
    ]

    all_cols = display_cols + [c for c in metric_cols + param_cols if c in runs.columns]
    subset   = runs[all_cols].head(args.top)

    # Pretty rename
    rename = {
        "run_id":                                    "Run ID",
        "tags.mlflow.runName":                       "Run Name",
        f"metrics.{args.metric}":                    args.metric.replace("_", " ").title(),
        "metrics.test_accuracy":                     "Test Acc",
        "metrics.test_top3_categorical_accuracy":    "Top-3 Acc",
        "metrics.macro_f1":                          "Macro F1",
        "metrics.test_roc_auc_macro":                "ROC-AUC",
        "params.architecture":                       "Architecture",
        "params.epochs":                             "Epochs",
        "params.batch_size":                         "Batch",
        "params.learning_rate":                      "LR",
    }
    subset = subset.rename(columns={k: v for k, v in rename.items() if k in subset.columns})

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n{'='*120}")
    print(f"  Top {args.top} runs  (ranked by {args.metric})")
    print(f"{'='*120}")
    print(subset.to_string(index=False))
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
