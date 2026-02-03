#!/bin/bash

# Multi-Agent Chat Threading System - Test Runner
# Convenient wrapper for running different test suites

set -e

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_usage() {
    cat << EOF
Usage: ./run_tests.sh [OPTION]

Test Runner for Multi-Agent Chat Threading System

OPTIONS:
    unit            Run unit tests only (fast)
    integration     Run integration tests
    performance     Run performance/stress tests
    e2e             Run end-to-end tests (requires running API)
    all             Run all tests (unit + integration + performance)
    coverage        Run tests with coverage report
    quick           Run quick tests (unit only, no coverage)
    help            Show this help message

EXAMPLES:
    ./run_tests.sh unit          # Fast unit tests
    ./run_tests.sh e2e           # Full E2E test with real API
    ./run_tests.sh coverage      # Generate coverage report
    ./run_tests.sh all           # Complete test suite

For more details, see tests/README.md
EOF
}

run_unit_tests() {
    echo -e "${BLUE}Running Unit Tests...${NC}"
    pytest tests/ -v -m "not slow" \
        --ignore=tests/test_integration.py \
        --ignore=tests/test_performance.py \
        --ignore=tests/test_api.py
}

run_integration_tests() {
    echo -e "${BLUE}Running Integration Tests...${NC}"
    pytest tests/test_integration.py tests/test_api.py -v
}

run_performance_tests() {
    echo -e "${BLUE}Running Performance Tests...${NC}"
    pytest tests/test_performance.py -v
}

run_e2e_tests() {
    echo -e "${BLUE}Running End-to-End Tests...${NC}"
    
    # Check if API is running
    if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  API is not running on port 8001${NC}"
        echo "Please start the API first: ./start.sh"
        exit 1
    fi
    
    ./tests/e2e_test.sh
    
    # Show log location
    LATEST_LOG=$(ls -t tests/logs/e2e_results_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo ""
        echo -e "${GREEN}📄 E2E Test log saved:${NC} $LATEST_LOG"
    fi
}

run_all_tests() {
    echo -e "${BLUE}Running All Tests (Unit + Integration + Performance)...${NC}"
    pytest tests/ -v
}

run_with_coverage() {
    echo -e "${BLUE}Running Tests with Coverage Report...${NC}"
    pytest tests/ --cov=src --cov-report=html --cov-report=term
    echo ""
    echo -e "${GREEN}📊 Coverage report generated:${NC} htmlcov/index.html"
    echo "Open with: open htmlcov/index.html"
}

run_quick_tests() {
    echo -e "${BLUE}Running Quick Tests (Unit only)...${NC}"
    pytest tests/ -v -x -m "not slow" \
        --ignore=tests/test_integration.py \
        --ignore=tests/test_performance.py \
        --ignore=tests/test_api.py
}

# Main logic
case "${1:-help}" in
    unit)
        run_unit_tests
        ;;
    integration)
        run_integration_tests
        ;;
    performance)
        run_performance_tests
        ;;
    e2e)
        run_e2e_tests
        ;;
    all)
        run_all_tests
        ;;
    coverage)
        run_with_coverage
        ;;
    quick)
        run_quick_tests
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Test run completed${NC}"

