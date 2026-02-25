#!/usr/bin/env bash
# ==============================================================================
# start_services.sh
# Quick launcher for MLflow + Airflow (Standalone Mode)
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AIRFLOW_HOME="${PROJECT_ROOT}/airflow_home"
LOGS_DIR="${PROJECT_ROOT}/logs"

# ── Activate Virtual Environment ─────────────────────────────────────────────
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_ROOT}/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOGS_DIR}"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║      CIFAR-10 MLOps – Starting Services          ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. MLflow tracking server (Port 5050) ────────────────────────────────────
echo -e "${GREEN}[1/2] Starting MLflow UI on http://localhost:5050 …${NC}"
mlflow ui \
  --backend-store-uri "file://${PROJECT_ROOT}/mlruns" \
  --default-artifact-root "file://${PROJECT_ROOT}/mlruns" \
  --port 5050 \
  --host 0.0.0.0 \
  > "${LOGS_DIR}/mlflow.log" 2>&1 &
MLFLOW_PID=$!
echo "      PID: ${MLFLOW_PID}  → logs: ${LOGS_DIR}/mlflow.log"

sleep 3

# ── 2. Airflow Standalone ───────────────────────────────────────────────────
# Airflow standalone starts everything: scheduler, triggerer, webserver.
echo -e "${GREEN}[2/2] Starting Airflow (Standalone Mode) on http://localhost:8080 …${NC}"
echo -e "      ${YELLOW}(This may take 15-30 seconds to initialize)${NC}"
airflow standalone \
  > "${LOGS_DIR}/airflow_standalone.log" 2>&1 &
AIRFLOW_PID=$!
echo "      PID: ${AIRFLOW_PID}  → logs: ${LOGS_DIR}/airflow_standalone.log"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  🚀 ${CYAN}MLflow UI${NC}     →  http://localhost:5050"
echo -e "  🌊 ${CYAN}Airflow UI${NC}    →  http://localhost:8080"
echo -e "     ${NC}Login:${NC}        Check ${LOGS_DIR}/airflow_standalone.log for password"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  To stop all services:"
echo "    kill ${MLFLOW_PID} ${AIRFLOW_PID}"
echo ""
echo "  To trigger the DAG immediately:"
echo "    airflow dags trigger cifar10_training_pipeline"
echo ""

# Keep alive to see background output if any
wait
