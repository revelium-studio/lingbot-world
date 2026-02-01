#!/bin/bash
# Start both backend and frontend for development

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Starting LingBot-World development servers...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run scripts/setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start backend in background
echo -e "${GREEN}Starting backend server on port 8000...${NC}"
python -m backend.main &
BACKEND_PID=$!

# Give backend time to start
sleep 2

# Start frontend
echo -e "${GREEN}Starting frontend server on port 5173...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!

cd "$PROJECT_ROOT"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Development Servers Running               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop all servers"

# Trap to clean up on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Servers stopped."
}

trap cleanup EXIT

# Wait for processes
wait
