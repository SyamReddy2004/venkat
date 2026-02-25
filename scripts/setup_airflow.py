#!/usr/bin/env python3
"""
scripts/setup_airflow.py
-------------------------
Programmatically bootstrap Airflow for the CIFAR-10 pipeline:
  - Sets AIRFLOW_HOME to the project directory
  - Initialises the Airflow database
  - Creates an admin user
  - Sets the cifar10_accuracy_threshold Airflow Variable

Run once before starting the Airflow webserver / scheduler.

Usage
-----
    python scripts/setup_airflow.py
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AIRFLOW_HOME = PROJECT_ROOT / "airflow_home"

# Ensure AIRFLOW_HOME is set before importing airflow
os.environ.setdefault("AIRFLOW_HOME", str(AIRFLOW_HOME))
os.environ.setdefault("AIRFLOW__CORE__DAGS_FOLDER",
                      str(PROJECT_ROOT / "dags"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
os.environ.setdefault("AIRFLOW__CORE__EXECUTOR", "SequentialExecutor")

AIRFLOW_HOME.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def run(cmd: str) -> int:
    log.info("$ %s", cmd)
    return subprocess.call(cmd, shell=True, env=os.environ)


def main():
    # 1. Init DB
    log.info("Initialising Airflow database …")
    rc = run("airflow db init")
    if rc != 0:
        log.error("airflow db init failed (exit %d)", rc)
        sys.exit(rc)

    # 2. Create admin user (ignore error if already exists)
    log.info("Creating admin user …")
    run(
        "airflow users create "
        "--username admin "
        "--password admin "
        "--firstname ML "
        "--lastname Engineer "
        "--role Admin "
        "--email admin@cifar10-mlops.local"
    )

    # 3. Set pipeline Variables
    log.info("Setting Airflow Variables …")
    run("airflow variables set cifar10_accuracy_threshold 0.70")

    # 4. Print next steps
    print("\n" + "=" * 64)
    print("  Airflow setup complete!")
    print("=" * 64)
    print(f"  AIRFLOW_HOME  : {AIRFLOW_HOME}")
    print(f"  DAGs folder   : {PROJECT_ROOT / 'dags'}")
    print()
    print("  Start the scheduler (in a separate terminal):")
    print(f"    export AIRFLOW_HOME={AIRFLOW_HOME}")
    print(f"    export AIRFLOW__CORE__DAGS_FOLDER={PROJECT_ROOT / 'dags'}")
    print( "    airflow scheduler")
    print()
    print("  Start the web UI (in another terminal):")
    print(f"    export AIRFLOW_HOME={AIRFLOW_HOME}")
    print( "    airflow webserver --port 8080")
    print()
    print("  Then open: http://localhost:8080  (admin / admin)")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
