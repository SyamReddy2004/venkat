#!/usr/bin/env python3
"""
scripts/run_training.py
------------------------
Standalone script to run the full training pipeline outside of Airflow.
Useful for quick local experiments.

Usage
-----
    python scripts/run_training.py [--arch resnet_custom] [--epochs 30]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.data_preprocessing import load_cifar10, preprocess_arrays, build_tf_datasets
from src.model import build_model
from src.train import train
from src.model_registry import compare_and_promote

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training Script")
    parser.add_argument("--arch",     default=None,
                        choices=["simple_cnn", "resnet_custom", "efficientnet_transfer"],
                        help="Model architecture (overrides config.yaml)")
    parser.add_argument("--epochs",   type=int,   default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch",    type=int,   default=None,
                        help="Batch size")
    parser.add_argument("--lr",       type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--register", action="store_true",
                        help="Register and promote model after training")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Minimum accuracy for model registration")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()

    # Apply CLI overrides
    if args.arch:   cfg["model"]["architecture"]         = args.arch
    if args.epochs: cfg["training"]["epochs"]            = args.epochs
    if args.batch:  cfg["training"]["batch_size"]        = args.batch
    if args.lr:     cfg["training"]["learning_rate"]     = args.lr
    if args.no_augment: cfg["data"]["augment"]           = False

    log.info("=" * 60)
    log.info("  CIFAR-10 Training Run")
    log.info("  Architecture : %s", cfg["model"]["architecture"])
    log.info("  Epochs       : %d", cfg["training"]["epochs"])
    log.info("  Batch size   : %d", cfg["training"]["batch_size"])
    log.info("  Learning rate: %.4f", cfg["training"]["learning_rate"])
    log.info("  Augmentation : %s", cfg["data"]["augment"])
    log.info("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    data_cfg     = cfg["data"]
    training_cfg = cfg["training"]

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

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model     = build_model(
        architecture  = model_cfg["architecture"],
        input_shape   = tuple(model_cfg["input_shape"]),
        num_classes   = model_cfg["num_classes"],
        dropout_rate  = model_cfg["dropout_rate"],
        learning_rate = training_cfg["learning_rate"],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    results = train(model, train_ds, val_ds, test_ds, cfg)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print("=" * 60)

    # ── Optional model registration ───────────────────────────────────────
    if args.register and results.get("run_id"):
        log.info("Registering model …")
        reg_result = compare_and_promote(
            cfg               = cfg,
            run_id            = results["run_id"],
            metric_key        = "test_accuracy",
            minimum_threshold = args.threshold,
        )
        log.info("Registration result: %s", reg_result)


if __name__ == "__main__":
    main()
