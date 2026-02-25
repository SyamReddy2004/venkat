#!/usr/bin/env bash
# ==============================================================================
# launch_all.sh — The One-Click CIFAR-10 MLOps Launcher
# ==============================================================================
# This script bundles everything: environment activation, service startup, 
# and provides direct links to the interfaces.
# ==============================================================================

PROJECT_ROOT="/Users/thurakapaulson/.gemini/antigravity/scratch/cifar10-mlops"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow_home"

# Colours for the terminal
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN} 🧠 CIFAR-10 MLOPS MASTER LAUNCHER ${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "${PROJECT_ROOT}"

# 1. Check Virtual Environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[!] Virtual environment not found. Setting up...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
else
    source .venv/bin/activate
fi

# 2. Setup Airflow & MLflow (if not already done)
if [ ! -f "airflow_home/airflow.db" ]; then
    echo -e "${YELLOW}[!] Initializing project databases...${NC}"
    python3 scripts/setup_mlflow.py
    python3 scripts/setup_airflow.py
fi

# 3. Kill any existing instances to avoid port conflicts
echo -e "${YELLOW}[*] Refreshing services...${NC}"
pkill -f "mlflow ui" || true
pkill -f "airflow scheduler" || true
pkill -f "airflow webserver" || true
sleep 1

# 4. Start Services using the existing script
bash scripts/start_services.sh &
LAUNCHER_PID=$!

# 5. Final Output
echo -e "${GREEN}✅ SERVICES ARE STARTING IN THE BACKGROUND!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e " 🚀 ${YELLOW}MLflow UI:${NC}    http://localhost:5000"
echo -e " 🌊 ${YELLOW}Airflow UI:${NC}   http://localhost:8080"
echo -e "    ${NC}Login:${NC}        admin / admin"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e " ${GREEN}To trigger a new training run now, paste this command:${NC}"
echo -e " ${BOLD}airflow dags trigger cifar10_training_pipeline${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Press Ctrl+C to stop the launcher (services will keep running in background)"

# Wait for background services to stabilize or user to exit
wait
