#!/bin/bash

echo "=========================================="
echo "Starting Agro-Suggester Services Locally"
echo "Press Ctrl+C to stop all services"
echo "==========================================\n"

# Ensures all background processes are killed when you hit Ctrl+C
trap 'echo -e "\nStopping all services..."; kill $(jobs -p) 2>/dev/null; exit' EXIT INT TERM

echo "[0/5] Setting up Virtual Environment..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# 1. Recommendation Service
echo "[1/5] Setting up Recommendation Service..."
cd recommendation
pip install -r requirements.txt -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 > ../recommendation.log 2>&1 &
cd ..
echo "      Recommendation Service running on port 8002 (logs -> recommendation.log)"

# 2. Input Prep Service
echo "[2/5] Setting up Input Prep Service..."
cd input_prep
pip install -r requirements.txt -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 > ../input_prep.log 2>&1 &
cd ..
echo "      Input Prep Service running on port 8001 (logs -> input_prep.log)"

# 3. API Gateway
echo "[3/5] Setting up API Gateway..."
cd gateway
pip install -r requirements.txt -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../gateway.log 2>&1 &
cd ..
echo "      API Gateway running on port 8000 (logs -> gateway.log)"

# 4. React Frontend
echo "[4/5] Setting up Frontend..."
cd frontend
npm install --silent
echo "      Frontend starting on http://localhost:5173"
echo "=========================================="

# Run frontend in the foreground
export VITE_API_GATEWAY=http://localhost:8000
npm run dev
