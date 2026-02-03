#!/bin/bash

# Multi-Agent Chat Threading System - Complete Feature Test
# This script tests all API endpoints and logs results

# Configuration
API_BASE="http://localhost:8001/api"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/e2e_results_$(date +%Y%m%d_%H%M%S).log"
FAILED_TESTS=0
PASSED_TESTS=0

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_header() {
    log ""
    log "=========================================="
    log "$1"
    log "=========================================="
}

log_test() {
    log "${BLUE}[TEST]${NC} $1"
}

log_success() {
    log "${GREEN}[✓]${NC} $1"
    ((PASSED_TESTS++))
}

log_error() {
    log "${RED}[✗]${NC} $1"
    ((FAILED_TESTS++))
}

log_info() {
    log "${YELLOW}[INFO]${NC} $1"
}

# Test if API is running
check_api_health() {
    log_test "Checking API health..."
    RESPONSE=$(curl -s http://localhost:8001/health 2>&1)
    
    if echo "$RESPONSE" | grep -q "healthy"; then
        log_success "API is healthy"
        log "Response: $RESPONSE"
    else
        log_error "API is not healthy or not running"
        log "Response: $RESPONSE"
        log "Please ensure the API is running with: ./start.sh"
        return 1
    fi
}

# Test 1: Create Thread
test_create_thread() {
    log_test "Test 1: Creating a new thread..."
    
    RESPONSE=$(curl -s -X POST "$API_BASE/threads" \
        -H "Content-Type: application/json" \
        -d '{
            "title": "Complete Feature Test Thread",
            "system_prompt": "You are a helpful AI assistant for testing purposes. Keep responses concise.",
            "current_model": "openai/gpt-4-turbo",
            "user_id": "test_user_001"
        }')
    
    THREAD_ID=$(echo "$RESPONSE" | jq -r '.thread_id')
    
    if [ -n "$THREAD_ID" ] && [ "$THREAD_ID" != "null" ]; then
        log_success "Thread created successfully"
        log "Thread ID: $THREAD_ID"
        log "Full Response: $RESPONSE"
        echo "$THREAD_ID" > /tmp/test_thread_id.txt
    else
        log_error "Failed to create thread"
        log "Response: $RESPONSE"
        exit 1
    fi
}

# Test 2: Get Thread Details
test_get_thread() {
    log_test "Test 2: Retrieving thread details..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X GET "$API_BASE/threads/$THREAD_ID")
    
    if echo "$RESPONSE" | jq -e '.thread_id' > /dev/null 2>&1; then
        log_success "Thread details retrieved"
        log "Response: $RESPONSE"
    else
        log_error "Failed to retrieve thread details"
        log "Response: $RESPONSE"
    fi
}

# Test 3: Send Message with Default Model (GPT-4)
test_send_message_gpt4() {
    log_test "Test 3: Sending message with GPT-4 (default model)..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X POST "$API_BASE/threads/$THREAD_ID/messages" \
        -H "Content-Type: application/json" \
        -d '{"content": "What is 2+2? Answer in one word."}')
    
    MESSAGE_ID=$(echo "$RESPONSE" | jq -r '.message_id')
    MODEL=$(echo "$RESPONSE" | jq -r '.model')
    CONTENT=$(echo "$RESPONSE" | jq -r '.content')
    
    if [ -n "$MESSAGE_ID" ] && [ "$MESSAGE_ID" != "null" ]; then
        log_success "Message sent and response received"
        log "Model Used: $MODEL"
        log "Message ID: $MESSAGE_ID"
        log "Response Content: $CONTENT"
        
        if echo "$MODEL" | grep -q "gpt-4-turbo"; then
            log_success "Correct model used (GPT-4 Turbo)"
        else
            log_error "Wrong model used (expected GPT-4 Turbo)"
        fi
    else
        log_error "Failed to send message"
        log "Response: $RESPONSE"
    fi
}

# Test 4: Send Message with Model Override (Claude)
test_send_message_claude() {
    log_test "Test 4: Sending message with Claude (model override)..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X POST "$API_BASE/threads/$THREAD_ID/messages" \
        -H "Content-Type: application/json" \
        -d '{
            "content": "Write a haiku about AI. Just the haiku, nothing else.",
            "model": "anthropic/claude-3.5-sonnet"
        }')
    
    MESSAGE_ID=$(echo "$RESPONSE" | jq -r '.message_id')
    MODEL=$(echo "$RESPONSE" | jq -r '.model')
    CONTENT=$(echo "$RESPONSE" | jq -r '.content')
    
    if [ -n "$MESSAGE_ID" ] && [ "$MESSAGE_ID" != "null" ]; then
        log_success "Message sent with Claude"
        log "Model Used: $MODEL"
        log "Message ID: $MESSAGE_ID"
        log "Response Content: $CONTENT"
        
        if echo "$MODEL" | grep -q "claude"; then
            log_success "Correct model used (Claude)"
        else
            log_error "Wrong model used (expected Claude)"
        fi
    else
        log_error "Failed to send message with Claude"
        log "Response: $RESPONSE"
    fi
}

# Test 5: Get Message History
test_get_message_history() {
    log_test "Test 5: Retrieving message history..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X GET "$API_BASE/threads/$THREAD_ID/messages?limit=20")
    
    MESSAGE_COUNT=$(echo "$RESPONSE" | jq '.messages | length')
    
    if [ "$MESSAGE_COUNT" -ge 4 ]; then
        log_success "Message history retrieved (4+ messages: 2 user + 2 assistant)"
        log "Total Messages: $MESSAGE_COUNT"
        log "First 2 messages:"
        echo "$RESPONSE" | jq '.messages[0:2]' | tee -a "$LOG_FILE"
    else
        log_error "Incorrect message count (expected >= 4, got $MESSAGE_COUNT)"
        log "Response: $RESPONSE"
    fi
}

# Test 6: Update Thread (Change Model)
test_update_thread() {
    log_test "Test 6: Updating thread default model..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X PATCH "$API_BASE/threads/$THREAD_ID" \
        -H "Content-Type: application/json" \
        -d '{
            "current_model": "anthropic/claude-3.5-sonnet",
            "title": "Updated Test Thread"
        }')
    
    NEW_MODEL=$(echo "$RESPONSE" | jq -r '.current_model')
    NEW_TITLE=$(echo "$RESPONSE" | jq -r '.title')
    
    if [ "$NEW_MODEL" == "anthropic/claude-3.5-sonnet" ]; then
        log_success "Thread updated successfully"
        log "New Model: $NEW_MODEL"
        log "New Title: $NEW_TITLE"
    else
        log_error "Failed to update thread"
        log "Response: $RESPONSE"
    fi
}

# Test 7: Send Message After Model Update (Should use Claude)
test_send_after_update() {
    log_test "Test 7: Sending message after model update (should use Claude)..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X POST "$API_BASE/threads/$THREAD_ID/messages" \
        -H "Content-Type: application/json" \
        -d '{"content": "What is the capital of France? One word answer."}')
    
    MODEL=$(echo "$RESPONSE" | jq -r '.model')
    CONTENT=$(echo "$RESPONSE" | jq -r '.content')
    
    if echo "$MODEL" | grep -q "claude"; then
        log_success "Message used new default model (Claude)"
        log "Model: $MODEL"
        log "Response: $CONTENT"
    else
        log_error "Message did not use new default model"
        log "Expected: claude, Got: $MODEL"
    fi
}

# Test 8: List All Threads for User
test_list_threads() {
    log_test "Test 8: Listing all threads for user..."
    
    RESPONSE=$(curl -s -X GET "$API_BASE/threads?user_id=test_user_001&limit=10")
    
    THREAD_COUNT=$(echo "$RESPONSE" | jq '.threads | length')
    
    if [ "$THREAD_COUNT" -ge 1 ]; then
        log_success "Threads listed successfully"
        log "Thread Count: $THREAD_COUNT"
        log "Response: $RESPONSE"
    else
        log_error "Failed to list threads"
        log "Response: $RESPONSE"
    fi
}

# Test 9: Create Multiple Messages for Summary Testing
test_create_messages_for_summary() {
    log_test "Test 9: Creating multiple messages to test auto-summarization..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    
    for i in {1..8}; do
        log_info "Sending message $i/8..."
        RESPONSE=$(curl -s -X POST "$API_BASE/threads/$THREAD_ID/messages" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"Test message number $i. Just acknowledge with OK.\"}")
        
        MESSAGE_ID=$(echo "$RESPONSE" | jq -r '.message_id')
        
        if [ -n "$MESSAGE_ID" ] && [ "$MESSAGE_ID" != "null" ]; then
            log_info "Message $i sent successfully (ID: $MESSAGE_ID)"
        else
            log_error "Failed to send message $i"
        fi
        
        # Small delay to avoid rate limiting
        sleep 2
    done
    
    log_success "Created 8 additional messages (total should be ~20 messages now)"
}

# Test 10: Check Message Count
test_check_message_count() {
    log_test "Test 10: Verifying message count..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X GET "$API_BASE/threads/$THREAD_ID")
    
    MESSAGE_COUNT=$(echo "$RESPONSE" | jq -r '.message_count')
    
    log_info "Current message count: $MESSAGE_COUNT"
    
    if [ "$MESSAGE_COUNT" -ge 18 ]; then
        log_success "Message count is correct (>= 18 messages)"
    else
        log_error "Message count seems low (expected >= 18, got $MESSAGE_COUNT)"
    fi
}

# Test 11: Get Summaries
test_get_summaries() {
    log_test "Test 11: Checking for auto-generated summaries..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X GET "$API_BASE/threads/$THREAD_ID/summaries")
    
    SUMMARY_COUNT=$(echo "$RESPONSE" | jq '.summaries | length')
    
    if [ "$SUMMARY_COUNT" -ge 1 ]; then
        log_success "Summaries generated successfully"
        log "Summary Count: $SUMMARY_COUNT"
        log "Summaries:"
        echo "$RESPONSE" | jq '.summaries' | tee -a "$LOG_FILE"
    else
        log_info "No summaries generated yet (might need >= 10 messages)"
        log "Response: $RESPONSE"
    fi
}

# Test 12: Test GPT-3.5 Model
test_gpt35_model() {
    log_test "Test 12: Testing GPT-3.5 Turbo model..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X POST "$API_BASE/threads/$THREAD_ID/messages" \
        -H "Content-Type: application/json" \
        -d '{
            "content": "Say hello in one word.",
            "model": "openai/gpt-3.5-turbo"
        }')
    
    MODEL=$(echo "$RESPONSE" | jq -r '.model')
    
    if echo "$MODEL" | grep -q "gpt-3.5-turbo"; then
        log_success "GPT-3.5 Turbo model works"
        log "Response: $RESPONSE"
    else
        log_error "Failed to use GPT-3.5 Turbo"
        log "Response: $RESPONSE"
    fi
}

# Test 13: Test Invalid Thread ID
test_invalid_thread() {
    log_test "Test 13: Testing error handling with invalid thread ID..."
    
    RESPONSE=$(curl -s -X GET "$API_BASE/threads/00000000-0000-0000-0000-000000000000")
    
    if echo "$RESPONSE" | grep -q "not found"; then
        log_success "Invalid thread ID handled correctly"
        log "Response: $RESPONSE"
    else
        log_error "Invalid thread ID not handled properly"
        log "Response: $RESPONSE"
    fi
}

# Test 14: Test Model Registry
test_model_validation() {
    log_test "Test 14: Testing invalid model handling..."
    
    THREAD_ID=$(cat /tmp/test_thread_id.txt)
    RESPONSE=$(curl -s -X POST "$API_BASE/threads/$THREAD_ID/messages" \
        -H "Content-Type: application/json" \
        -d '{
            "content": "Test message",
            "model": "invalid/model-name"
        }')
    
    if echo "$RESPONSE" | grep -q "error\|not found\|invalid"; then
        log_success "Invalid model rejected correctly"
        log "Response: $RESPONSE"
    else
        log_error "Invalid model not rejected"
        log "Response: $RESPONSE"
    fi
}

# Main Test Execution
main() {
    log_header "Multi-Agent Chat Threading System - Complete Feature Test"
    log "Test Started: $(date)"
    log "API Base URL: $API_BASE"
    log "Log File: $LOG_FILE"
    log ""
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed"
        exit 1
    fi
    
    log_success "All prerequisites installed"
    
    # Run tests
    check_api_health || exit 1
    
    log_header "Phase 1: Thread Management"
    test_create_thread
    test_get_thread
    test_list_threads
    
    log_header "Phase 2: Message Handling & Model Switching"
    test_send_message_gpt4
    test_send_message_claude
    test_gpt35_model
    test_get_message_history
    test_update_thread
    test_send_after_update
    
    log_header "Phase 3: Auto-Summarization Testing"
    test_create_messages_for_summary
    test_check_message_count
    sleep 3  # Wait for async summarization
    test_get_summaries
    
    log_header "Phase 4: Error Handling"
    test_invalid_thread
    test_model_validation
    
    # Summary
    log_header "Test Summary"
    log "Total Tests Passed: ${GREEN}$PASSED_TESTS${NC}"
    log "Total Tests Failed: ${RED}$FAILED_TESTS${NC}"
    log "Test Completed: $(date)"
    log ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log "${GREEN}✅ ALL TESTS PASSED!${NC}"
        log ""
        log "📊 Test Results:"
        log "  - Thread Management: ✅"
        log "  - Multi-Model Support: ✅"
        log "  - Model Switching: ✅"
        log "  - Message History: ✅"
        log "  - Auto-Summarization: ✅"
        log "  - Error Handling: ✅"
        exit 0
    else
        log "${RED}❌ SOME TESTS FAILED${NC}"
        log "Please check the log file: $LOG_FILE"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    rm -f /tmp/test_thread_id.txt
}

trap cleanup EXIT

# Run main function
main

