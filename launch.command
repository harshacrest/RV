#!/bin/bash
# RV Dashboard Launcher
# Double-click this file to start both the API server and frontend

cd "$(dirname "$0")"

echo "=== Starting RV Dashboard ==="
echo ""

# Kill any existing processes on ports 5500 and 5501
lsof -ti:5500 | xargs kill -9 2>/dev/null
lsof -ti:5501 | xargs kill -9 2>/dev/null

# Start API server in background
echo "[1/2] Starting API server on port 5501..."
/opt/homebrew/bin/python3 api_server.py &
API_PID=$!
sleep 2

# Start Next.js frontend
echo "[2/2] Starting frontend on port 5500..."
cd rv-dashboard
npm run dev &
NEXT_PID=$!

echo ""
echo "==================================="
echo "  RV Dashboard is starting up!"
echo "  Frontend: http://localhost:5500"
echo "  API:      http://localhost:5501"
echo "==================================="
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait for Ctrl+C
trap "echo ''; echo 'Shutting down...'; kill $API_PID $NEXT_PID 2>/dev/null; exit 0" INT TERM
wait
