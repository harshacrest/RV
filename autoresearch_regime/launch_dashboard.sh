#!/bin/bash
# Launch the Autoresearch Dashboard (API + Frontend)
# Usage: ./launch_dashboard.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null
        echo "  API server stopped"
    fi
    exit 0
}

trap cleanup INT TERM

echo "====================================="
echo "  Autoresearch Dashboard Launcher"
echo "====================================="
echo ""

# Check dependencies
if ! command -v python &>/dev/null; then
    echo "ERROR: python not found"
    exit 1
fi

if ! command -v npm &>/dev/null; then
    echo "ERROR: npm not found"
    exit 1
fi

# Check if diagnostics data exists
if [ ! -f "$SCRIPT_DIR/output/diagnostics_report.json" ]; then
    echo "No diagnostics data found. Running diagnostics..."
    echo ""
    cd "$SCRIPT_DIR"
    python diagnostics.py --save
    echo ""
fi

# Install dashboard deps if needed
if [ ! -d "$SCRIPT_DIR/dashboard/node_modules" ]; then
    echo "Installing dashboard dependencies..."
    cd "$SCRIPT_DIR/dashboard"
    npm install
    echo ""
fi

# Check if port 5502 is already in use
if lsof -i :5502 &>/dev/null; then
    echo "Port 5502 already in use — killing existing process"
    lsof -ti :5502 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Start API server in background
echo "Starting API server on port 5502..."
cd "$SCRIPT_DIR"
python api_autoresearch.py &
API_PID=$!
sleep 2

# Verify API is up
if ! curl -s http://localhost:5502/api/baseline >/dev/null 2>&1; then
    echo "WARNING: API server may not be ready yet"
fi

echo "Starting dashboard on port 5503..."
echo ""
echo "  API:       http://localhost:5502"
echo "  Dashboard: http://localhost:5503"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "====================================="
echo ""

# Start frontend (blocking — keeps script alive)
cd "$SCRIPT_DIR/dashboard"
PORT=5503 npm run dev
