#!/usr/bin/env bash
# ==============================================================================
# MASTER_RUN.sh — One-Click Activation & Pipeline Trigger
# ==============================================================================

PROJECT_ROOT="/Users/thurakapaulson/.gemini/antigravity/scratch/cifar10-mlops"
export AIRFLOW_HOME="${PROJECT_ROOT}/airflow_home"

# Colours
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "${PROJECT_ROOT}"

echo -e "${CYAN}🚀 Initializing CIFAR-10 MLOps Pipeline...${NC}"

# 1. Activate Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo -e "${RED}Error: .venv not found. Please run ./launch_all.sh first.${NC}"
    exit 1
fi

# 2. Ensure Services are started
# Check if MLflow is running on 5050
if ! lsof -i:5050 > /dev/null; then
    echo -e "${YELLOW}[!] Services not detected. Starting them now...${NC}"
    bash scripts/start_services.sh &
    sleep 5
fi

# 3. Trigger the DAG
echo -e "${GREEN}[+] Triggering Airflow Pipeline...${NC}"
airflow dags trigger cifar10_training_pipeline

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e " ✅ ${GREEN}PIPELINE TRIGGERED SUCCESSFULLY!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e " 📊 ${YELLOW}Track Live at:${NC}   http://localhost:8080"
echo -e " 📈 ${YELLOW}Results at:${NC}      http://localhost:5050"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e " ${YELLOW}Tip:${NC} Keep this terminal open while services start up."
