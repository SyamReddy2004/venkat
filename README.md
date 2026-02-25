# 🧠 CIFAR-10 MLOps Pipeline

> **Multiclass image classification on CIFAR-10 with TensorFlow — experiment
> tracking with MLflow — automated training orchestration with Apache Airflow.**

---

## 🗂 Project Structure

```
cifar10-mlops/
├── config/
│   └── config.yaml              # Central configuration (data, model, training, MLflow, Airflow)
├── src/
│   ├── __init__.py
│   ├── config_loader.py         # YAML config loader
│   ├── data_preprocessing.py    # CIFAR-10 loading, normalisation, augmentation, tf.data pipelines
│   ├── model.py                 # 3 architectures + callback factory
│   ├── train.py                 # Training loop with full MLflow tracking
│   ├── evaluate.py              # Comprehensive evaluation (ROC, PR curves, confusion matrix)
│   └── model_registry.py       # Register, compare, and promote models
├── dags/
│   └── cifar10_training_pipeline.py   # Airflow DAG (7 tasks, TaskFlow API)
├── scripts/
│   ├── run_training.py          # Standalone CLI training
│   ├── run_evaluation.py        # Standalone CLI evaluation
│   ├── setup_mlflow.py          # One-time MLflow initialisation
│   ├── setup_airflow.py         # One-time Airflow bootstrap
│   ├── compare_runs.py          # Compare MLflow runs
│   └── start_services.sh        # Launch all services
├── tests/
│   └── test_pipeline.py         # pytest suite (config, data, model, training)
├── models/                      # Saved model checkpoints (git-ignored)
├── logs/                        # TensorBoard logs + evaluation plots (git-ignored)
├── mlruns/                      # MLflow artefact store (git-ignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### 1 — Create a virtual environment

```bash
cd cifar10-mlops
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows
```

### 2 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon (M-series)** — install the Metal-optimised TensorFlow instead:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

### 3 — Initialize MLflow

```bash
python scripts/setup_mlflow.py
```

### 4 — Start MLflow UI (keep this terminal open)

```bash
mlflow ui --port 5000 --backend-store-uri file://$(pwd)/mlruns
# → open http://localhost:5000
```

### 5 — Run training (standalone, fastest way to test)

```bash
python scripts/run_training.py
# or with overrides:
python scripts/run_training.py --arch simple_cnn --epochs 5 --register
```

---

## 🏗 Model Architectures

| Key | Description | Params |
|-----|-------------|--------|
| `simple_cnn` | 3-block VGG-style CNN (fast baseline) | ~1.2 M |
| `resnet_custom` ⭐ | 4-stage custom ResNet with pre-activation residual blocks | ~6.6 M |
| `efficientnet_transfer` | EfficientNetB0 (frozen) + custom head, fine-tunable | ~4.3 M |

Switch architecture in `config/config.yaml`:

```yaml
model:
  architecture: resnet_custom   # simple_cnn | resnet_custom | efficientnet_transfer
```

---

## 📊 What Gets Tracked in MLflow

Every training run logs:

| Category | Items |
|----------|-------|
| **Parameters** | architecture, epochs, batch size, LR, dropout, optimizer, augmentation, … |
| **Metrics (per epoch)** | loss, accuracy, val_loss, val_accuracy, top-3 accuracy, AUC |
| **Test metrics** | accuracy, top-3 accuracy, AUC, macro F1, weighted F1, per-class F1/P/R |
| **Artefacts** | model summary TXT, confusion matrix PNG, per-class metrics bar chart, training curves PNG, ROC curves PNG, precision-recall curves PNG, confidence histogram PNG, classification report TXT |
| **Model** | Keras model → MLflow Model Registry (`cifar10-resnet`) |

---

## 🔄 Airflow DAG

The DAG `cifar10_training_pipeline` (in `dags/`) has **7 tasks** wired with the
TaskFlow API:

```
check_environment
      │
preprocess_data
      │
train_model           ← MLflow run starts here
      │
evaluate_model        ← Loads model from the run's MLflow artefact
      │
register_model        ← Compare vs Production; promote if better
      │
generate_report       ← Self-contained HTML + JSON report saved to disk
      │
notify                ← Prints summary to Airflow task logs
```

### XCom data flow

| Producer | → Consumer | Payload |
|----------|-----------|---------|
| `check_environment` | `preprocess_data` | env info dict |
| `preprocess_data` | `train_model` | dataset stats |
| `train_model` | `evaluate_model`, `register_model`, `generate_report`, `notify` | run_id, metrics |
| `evaluate_model` | `register_model`, `generate_report`, `notify` | full metrics dict |
| `register_model` | `generate_report`, `notify` | version, promoted flag |
| `generate_report` | `notify` | HTML report path |

### Quality gate

Model registration requires `test_accuracy ≥ 0.70` (overridable at runtime
via Airflow Variable `cifar10_accuracy_threshold`).

---

## ⚙️  Airflow Setup & Run

### Setup (once)

```bash
python scripts/setup_airflow.py
```

### Start all services

```bash
bash scripts/start_services.sh
# MLflow UI  →  http://localhost:5000
# Airflow UI →  http://localhost:8080  (admin / admin)
```

Or manually in separate terminals:

```bash
# Terminal 1 — MLflow
mlflow ui --port 5000

# Terminal 2 — Airflow scheduler
export AIRFLOW_HOME="$(pwd)/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow scheduler

# Terminal 3 — Airflow webserver
export AIRFLOW_HOME="$(pwd)/airflow_home"
airflow webserver --port 8080
```

### Trigger the DAG

```bash
# From CLI
airflow dags trigger cifar10_training_pipeline

# Or via Airflow UI → http://localhost:8080 → DAGs → cifar10_training_pipeline → ▶ Trigger
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# with coverage:
pytest tests/ -v --cov=src --cov-report=term-missing
```

The test suite covers:

- ✅ Config loading & validation
- ✅ Data split sizes, shapes, one-hot encoding, no data leakage
- ✅ Pixel range after normalisation & augmentation shape preservation
- ✅ Model output shapes & softmax constraint (rows sum to 1)
- ✅ All three architectures build without errors
- ✅ Callback creation for all LR scheduler types
- ✅ 2-epoch mini-training smoke test
- ✅ Residual block dimension arithmetic

---

## 📈 Training Configuration Reference

Edit `config/config.yaml` to tune the pipeline:

```yaml
training:
  epochs: 30            # total epochs (early stopping may cut short)
  batch_size: 64
  learning_rate: 0.001
  lr_scheduler: "cosine"      # cosine | step | constant
  early_stopping_patience: 7
  optimizer: "adam"

data:
  validation_split: 0.1       # carved out of training set
  augment: true               # random crop, flip, brightness, contrast, saturation
  random_seed: 42
```

---

## 🔗 Comparing Runs

```bash
# Top 10 runs by accuracy
python scripts/compare_runs.py

# Top 5 runs sorted by macro F1
python scripts/compare_runs.py --top 5 --metric macro_f1
```

---

## 📦 Model Registry Workflow

```
New run (test_accuracy) ──► < threshold? ──► Rejected (not registered)
                               │
                            ≥ threshold
                               │
                         Register version
                               │
                      Better than Production?
                         │               │
                        Yes              No
                         │               │
                    Promote to      Keep in Staging
                    Production      (archive old)
```

---

## 🎯 Expected Results

| Architecture | Epochs | Test Accuracy | Macro F1 |
|---|---|---|---|
| `simple_cnn` | 30 | ~82–85% | ~0.82 |
| `resnet_custom` ⭐ | 30 | **~87–91%** | **~0.88** |
| `efficientnet_transfer` | 30 (frozen) | ~85–88% | ~0.86 |

> Results vary with random seed, hardware, and exact TensorFlow version.

---

## 📁 Output Artefacts (per run)

```
logs/<run_name>/
├── best_model.keras             # Best checkpoint (val_accuracy)
├── training_curves.png          # Accuracy + loss over epochs
├── confusion_matrix.png         # Normalised confusion matrix
├── per_class_metrics.png        # Precision / Recall / F1 bar chart
├── classification_report.txt    # sklearn full report
└── evaluation/
    ├── roc_curves.png           # Per-class ROC (one-vs-rest)
    ├── precision_recall_curves.png
    ├── confidence_histogram.png
    ├── per_class_metrics.csv
    └── metrics.json

logs/reports/
└── report_<run_id>.html         # Self-contained HTML performance report
```

---

## 🛠 Extending the Pipeline

| Goal | Where to edit |
|------|--------------|
| Add a new architecture | `src/model.py` → add builder function + register in `builders` dict |
| Change augmentation | `src/data_preprocessing.py` → `_augment()` |
| Add new tracked metrics | `src/train.py` → `_log_detailed_metrics()` |
| Add a new DAG task | `dags/cifar10_training_pipeline.py` → add `@task` and wire into flow |
| Change promotion threshold | Airflow Variable `cifar10_accuracy_threshold` (no code change needed) |
| Send Slack/email notification | `dags/.py` → `notify()` task |
