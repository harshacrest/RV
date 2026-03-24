#!/bin/bash
# RV Dashboard – Full Rebuild & Launch
# Double-click this file to regenerate all data and start the dashboard

cd "$(dirname "$0")"

echo "============================================"
echo "  RV Dashboard – Full Rebuild & Launch"
echo "============================================"
echo ""

# Kill any existing processes on ports 5500 and 5501
lsof -ti:5500 | xargs kill -9 2>/dev/null
lsof -ti:5501 | xargs kill -9 2>/dev/null

PYTHON=/opt/homebrew/bin/python3

# ── Step 1: Regenerate nifty_spot_daily.parquet ──
echo "[1/5] Fetching & aggregating spot data..."
$PYTHON fetch_data.py
if [ $? -ne 0 ]; then echo "FAILED: fetch_data.py"; exit 1; fi
echo "      ✓ nifty_spot_daily.parquet"
echo ""

# ── Step 2: Regenerate rv_daily.parquet ──
echo "[2/5] Computing RV features..."
$PYTHON rv_features.py
if [ $? -ne 0 ]; then echo "FAILED: rv_features.py"; exit 1; fi
echo "      ✓ rv_daily.parquet"
echo ""

# ── Step 3: Regenerate daily_overview Excel ──
echo "[3/5] Building daily overview Excel..."
$PYTHON build_daily_overview.py
if [ $? -ne 0 ]; then echo "FAILED: build_daily_overview.py"; exit 1; fi
echo "      ✓ daily_overview_all_strategies.xlsx"
echo ""

# ── Step 4: Start API server ──
echo "[4/5] Starting API server on port 5501..."
$PYTHON api_server.py &
API_PID=$!
sleep 2

# ── Step 5: Start frontend ──
echo "[5/5] Starting frontend on port 5500..."
cd rv-dashboard
npm run dev &
NEXT_PID=$!
cd ..

echo ""
echo "============================================"
echo "  All data rebuilt. Dashboard is starting!"
echo "  Frontend: http://localhost:5500"
echo "  API:      http://localhost:5501"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait for Ctrl+C
trap "echo ''; echo 'Shutting down...'; kill $API_PID $NEXT_PID 2>/dev/null; exit 0" INT TERM
wait
