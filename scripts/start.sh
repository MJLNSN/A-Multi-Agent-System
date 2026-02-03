#!/bin/bash

# Multi-Agent Chat Threading System - Production Startup Script
# This script handles all setup, port cleanup, and service initialization

set -e

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Multi-Agent Chat Threading System"
echo "Production Startup Script"
echo "=========================================="
echo ""

# Configuration
API_PORT=${PORT:-8001}
DB_PORT=5433

# Step 1: Port Cleanup
echo "🔧 Step 1: Cleaning up ports..."
echo "  - Checking API port $API_PORT..."
if lsof -ti:$API_PORT > /dev/null 2>&1; then
    echo "  - Killing process on port $API_PORT..."
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi
echo "  ✅ Port $API_PORT is clear"
echo ""

# Step 2: Start PostgreSQL
echo "🗄️  Step 2: Starting PostgreSQL..."
echo "  - Stopping old containers..."
docker-compose down 2>/dev/null || true

echo "  - Starting PostgreSQL on port $DB_PORT..."
docker-compose up -d db

echo "  - Waiting for database to be ready..."
for i in {1..30}; do
    if docker-compose exec -T db psql -U chatuser -d chatdb -c "SELECT 1" > /dev/null 2>&1; then
        echo "  ✅ PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ❌ PostgreSQL failed to start after 30 seconds"
        echo "  Check logs: docker-compose logs db"
        exit 1
    fi
    sleep 1
done
echo ""

# Step 3: Check environment
echo "🔍 Step 3: Verifying environment..."
if [ ! -f .env ]; then
    echo "  ⚠️  .env file not found, creating from env.example..."
    cp env.example .env
    echo "  ⚠️  Please edit .env and add your OPENROUTER_API_KEY"
    echo "  Then run this script again."
    exit 1
fi

if grep -q "OPENROUTER_API_KEY=$" .env 2>/dev/null || grep -q "OPENROUTER_API_KEY=\"\"" .env 2>/dev/null; then
    echo "  ⚠️  OPENROUTER_API_KEY is not set in .env"
    echo "  Please edit .env and add your API key."
    exit 1
fi
echo "  ✅ Environment configured"
echo ""

# Step 4: Install dependencies (optional)
if [ "$1" == "--install" ]; then
    echo "📦 Step 4: Installing dependencies..."
    pip install -q -r requirements.txt
    echo "  ✅ Dependencies installed"
    echo ""
fi

# Step 5: Start API Service
echo "🚀 Step 5: Starting API service on port $API_PORT..."
echo "  - Starting uvicorn..."
PORT=$API_PORT uvicorn src.main:app --host 0.0.0.0 --port $API_PORT --reload > /tmp/api_server.log 2>&1 &
API_PID=$!
echo "  - Process ID: $API_PID"

echo "  - Waiting for service to be ready..."
for i in {1..20}; do
    if curl -s http://localhost:$API_PORT/docs | grep -q "Multi-Agent" 2>/dev/null; then
        echo "  ✅ API service is ready"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "  ❌ API service failed to start after 20 seconds"
        echo "  Check logs: tail -f /tmp/api_server.log"
        exit 1
    fi
    sleep 1
done
echo ""

# Step 6: Display status
echo "=========================================="
echo "✅ All services started successfully!"
echo "=========================================="
echo ""
echo "📊 Service Information:"
echo "  - API URL:        http://localhost:$API_PORT"
echo "  - Swagger Docs:   http://localhost:$API_PORT/docs"
echo "  - ReDoc:          http://localhost:$API_PORT/redoc"
echo "  - API PID:        $API_PID"
echo "  - Database Port:  $DB_PORT"
echo ""
echo "📝 Logs:"
echo "  - API:      tail -f /tmp/api_server.log"
echo "  - Database: docker-compose logs -f db"
echo ""
echo "🛑 Stop Services:"
echo "  - API:      kill $API_PID"
echo "  - Database: docker-compose down"
echo "  - All:      ./stop.sh"
echo ""
echo "🧪 Run Tests:"
echo "  - pytest tests/ -v"
echo ""
echo "🎯 Next Steps:"
echo "  1. Visit http://localhost:$API_PORT/docs to explore the API"
echo "  2. Create a thread and start chatting!"
echo ""
