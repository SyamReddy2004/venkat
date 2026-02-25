#!/usr/bin/env bash
# ==============================================================================
# start_services.sh
# Quick launcher for MLflow tracking server + Airflow scheduler + webserver.
# Run from the project root:  bash scripts/start_services.sh
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow_home"
LOGS_DIR="${PROJECT_ROOT}/logs"

# ── Activate Virtual Environment if it exists ───────────────────────────────
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

export AIRFLOW_HOME
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_ROOT}/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__EXECUTOR="SequentialExecutor"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOGS_DIR}"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║      CIFAR-10 MLOps – Starting Services          ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. MLflow tracking server ─────────────────────────────────────────────────
echo -e "${GREEN}[1/3] Starting MLflow UI on http://localhost:5050 …${NC}"
mlflow ui \
  --backend-store-uri "file://${PROJECT_ROOT}/mlruns" \
  --default-artifact-root "file://${PROJECT_ROOT}/mlruns" \
  --port 5050 \
  --host 0.0.0.0 \
  > "${LOGS_DIR}/mlflow.log" 2>&1 &
MLFLOW_PID=$!
echo "      PID: ${MLFLOW_PID}  → logs: ${LOGS_DIR}/mlflow.log"

sleep 2

# ── 2. Airflow scheduler ──────────────────────────────────────────────────────
echo -e "${GREEN}[2/3] Starting Airflow scheduler …${NC}"
airflow scheduler \
  > "${LOGS_DIR}/airflow_scheduler.log" 2>&1 &
SCHEDULER_PID=$!
echo "      PID: ${SCHEDULER_PID}  → logs: ${LOGS_DIR}/airflow_scheduler.log"

sleep 2

# ── 3. Airflow webserver ──────────────────────────────────────────────────────
echo -e "${GREEN}[3/3] Starting Airflow webserver on http://localhost:8080 …${NC}"
airflow webserver \
  --port 8080 \
  > "${LOGS_DIR}/airflow_webserver.log" 2>&1 &
WEBSERVER_PID=$!
echo "      PID: ${WEBSERVER_PID}  → logs: ${LOGS_DIR}/airflow_webserver.log"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${CYAN}MLflow UI${NC}        →  http://localhost:5000"
echo -e "  ${CYAN}Airflow UI${NC}       →  http://localhost:8080  (admin / admin)"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  To stop all services:"
echo "    kill ${MLFLOW_PID} ${SCHEDULER_PID} ${WEBSERVER_PID}"
echo ""
echo "  To trigger the DAG immediately:"
echo "    airflow dags trigger cifar10_training_pipeline"
echo ""

# Keep script alive so PIDs are visible; Ctrl+C to exit
wait
