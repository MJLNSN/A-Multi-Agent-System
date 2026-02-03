#!/bin/bash

# Multi-Agent Chat Threading System - Stop Script

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "🛑 Stopping Multi-Agent Chat Threading System..."
echo ""

# Stop API service
echo "Stopping API service..."
if lsof -ti:8001 > /dev/null 2>&1; then
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    echo "✅ API service stopped"
else
    echo "ℹ️  API service not running"
fi

# Stop database
echo ""
echo "Stopping PostgreSQL..."
docker-compose down 2>/dev/null
echo "✅ Database stopped"

echo ""
echo "✅ All services stopped"

