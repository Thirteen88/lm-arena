#!/bin/bash
# Super-Powered LM Arena Commands - Standalone Script

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Arena commands
arena-start() {
    print_info "Starting Super-Powered LM Arena..."
    ~/start-super-powered-arena.sh
}

arena-status() {
    print_info "Checking LM Arena status..."
    if curl -s http://localhost:8999/health > /dev/null; then
        print_status "LM Arena is running on port 8999"
        curl -s http://localhost:8999/health | jq . 2>/dev/null || curl -s http://localhost:8999/health
    else
        print_error "LM Arena is not running"
        return 1
    fi
}

arena-dashboard() {
    print_info "Opening monitoring dashboard..."
    if command -v xdg-open > /dev/null; then
        xdg-open http://localhost:8999/dashboard 2>/dev/null
    elif command -v open > /dev/null; then
        open http://localhost:8999/dashboard 2>/dev/null
    else
        echo "Dashboard available at: http://localhost:8999/dashboard"
    fi
}

arena-stop() {
    print_info "Stopping LM Arena..."
    if pkill -f "uvicorn.*api.main:app.*port 8999"; then
        print_status "LM Arena stopped"
    else
        print_warning "LM Arena was not running"
    fi
}

arena-restart() {
    arena-stop
    sleep 2
    arena-start
}

arena-logs() {
    print_info "Viewing LM Arena logs..."
    tail -f /tmp/lm-arena-api.log 2>/dev/null || print_error "Log file not found"
}

# Web automation testing
arena-test-search() {
    print_info "Testing web automation search..."
    curl -X POST http://localhost:8999/chat \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"search for weather in London\", \"model\": \"web-automation\", \"temperature\": 0.7, \"max_tokens\": 100, \"conversation_id\": \"test-$(date +%s)\"}"
}

arena-test-scrape() {
    print_info "Testing website scraping..."
    curl -X POST http://localhost:8999/chat \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"scrape https://example.com\", \"model\": \"web-automation\", \"temperature\": 0.7, \"max_tokens\": 100, \"conversation_id\": \"test-$(date +%s)\"}"
}

# Agent management
orchestrator-start() {
    print_info "Starting Claude Orchestrator..."
    cd /home/gary/claude-orchestrator && python3 orchestrator.py
}

orchestrator-status() {
    print_info "Checking Claude Orchestrator status..."
    ps aux | grep orchestrator | grep -v grep || print_warning "Claude Orchestrator not running"
}

manus-start() {
    print_info "Starting Manus automation agent..."
    cd /home/gary/manus-automation-agent && source venv/bin/activate && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
}

manus-status() {
    print_info "Checking Manus agent status..."
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Manus agent is running on port 8000"
    else
        print_warning "Manus agent is not running"
    fi
}

# Help function
arena-help() {
    echo "🤖 Super-Powered LM Arena - Command Reference"
    echo "=============================================="
    echo ""
    echo "🚀 Basic Commands:"
    echo "   arena-start     - Start the LM Arena system"
    echo "   arena-status    - Check system health status"
    echo "   arena-dashboard - Open monitoring dashboard"
    echo "   arena-stop      - Stop the LM Arena system"
    echo "   arena-restart   - Restart the LM Arena system"
    echo ""
    echo "📊 Monitoring:"
    echo "   arena-logs      - View real-time logs"
    echo "   arena-status    - Health check with JSON output"
    echo ""
    echo "🌐 Web Automation Testing:"
    echo "   arena-test-search - Test search automation"
    echo "   arena-test-scrape - Test website scraping"
    echo ""
    echo "🤖 Agent Management:"
    echo "   orchestrator-start - Start Claude Orchestrator"
    echo "   orchestrator-status - Check orchestrator status"
    echo "   manus-start      - Start Manus automation agent"
    echo "   manus-status     - Check Manus agent status"
    echo ""
    echo "📈 URLs:"
    echo "   Dashboard: http://localhost:8999/dashboard"
    echo "   API Docs:   http://localhost:8999/docs"
    echo "   Health:     http://localhost:8999/health"
    echo ""
}

# Main script execution
case "$1" in
    "start")
        arena-start
        ;;
    "status")
        arena-status
        ;;
    "dashboard")
        arena-dashboard
        ;;
    "stop")
        arena-stop
        ;;
    "restart")
        arena-restart
        ;;
    "logs")
        arena-logs
        ;;
    "test-search")
        arena-test-search
        ;;
    "test-scrape")
        arena-test-scrape
        ;;
    "help"|"-h"|"--help"|"")
        arena-help
        ;;
    *)
        print_error "Unknown command: $1"
        arena-help
        exit 1
        ;;
esac