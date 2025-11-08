#!/bin/bash

# LM Arena Deployment Script
# This script handles deployment and testing of the LM Arena framework

set -e  # Exit on any error

echo "🚀 LM Arena Deployment Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "core" ]; then
    print_error "Please run this script from the LM Arena root directory"
    exit 1
fi

print_status "Starting LM Arena deployment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
print_status "Installing dependencies..."
if pip install -r requirements.txt > /dev/null 2>&1; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Install LM Arena in development mode
print_status "Installing LM Arena package..."
if pip install -e . > /dev/null 2>&1; then
    print_success "LM Arena installed"
else
    print_warning "LM Arena installation had issues, continuing..."
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p data logs prompts models

# Copy configuration file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    print_status "Creating configuration file..."
    cp config.yaml.example config.yaml
    print_warning "Please edit config.yaml with your API keys and settings"
fi

# Validate Python syntax
print_status "Validating Python syntax..."
python_files=$(find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*")
syntax_errors=0

for file in $python_files; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file"
        ((syntax_errors++))
    fi
done

if [ $syntax_errors -eq 0 ]; then
    print_success "All Python files have valid syntax"
else
    print_error "Found $syntax_errors files with syntax errors"
    exit 1
fi

# Run basic functionality tests
print_status "Running basic functionality tests..."

# Test core imports
python -c "
import sys
sys.path.insert(0, '.')
try:
    from core.agent import LMArenaAgent
    from core.model_switcher import ModelSwitcher
    from prompts.prompt_manager import PromptManager
    print('✅ Core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "Core functionality test passed"
else
    print_warning "Core functionality test failed, but deployment continues"
fi

# Test API server startup (briefly)
print_status "Testing API server startup..."
timeout 5 python -m uvicorn api.main:app --host 127.0.0.1 --port 8999 > /dev/null 2>&1 &
API_PID=$!
sleep 2

if kill -0 $API_PID 2>/dev/null; then
    print_success "API server started successfully"
    kill $API_PID 2>/dev/null || true
else
    print_warning "API server test failed"
fi

# Git status check
if [ -d ".git" ]; then
    print_status "Git status:"
    git status --porcelain
    echo ""
fi

# Final summary
echo ""
print_success "🎉 LM Arena Deployment Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Edit config.yaml with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Start server: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "4. Open docs: http://localhost:8000/docs"
echo ""
echo "🔗 Useful Commands:"
echo "  - Start server: python main.py"
echo "  - Run tests: python -m pytest tests/ -v"
echo "  - Check health: curl http://localhost:8000/health"
echo "  - View models: curl http://localhost:8000/models"
echo ""

# Check for API keys
if [ -z "$LM_ARENA_OPENAI_API_KEY" ] && [ -z "$LM_ARENA_ANTHROPIC_API_KEY" ]; then
    print_warning "No API keys found in environment variables"
    echo "   Set your API keys:"
    echo "   export LM_ARENA_OPENAI_API_KEY='your-openai-key'"
    echo "   export LM_ARENA_ANTHROPIC_API_KEY='your-anthropic-key'"
fi

echo "🚀 Ready to launch LM Arena!"