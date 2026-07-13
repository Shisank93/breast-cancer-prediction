#!/bin/bash
echo "============================================="
echo "   Starting OncoPredict AI Web Application   "
echo "============================================="
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Check if python virtual env exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Launching FastAPI Uvicorn Server on http://127.0.0.1:8000..."
echo "Press Ctrl+C to stop the server."
echo ""

python -m uvicorn app.main:app --reload --port 8000
