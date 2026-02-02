#!/bin/bash
# UV Setup Script for RLM Document Retrieval System
# This script sets up the project using UV (fast Python package manager)

set -e  # Exit on error

echo "🚀 Setting up RLM Document Retrieval with UV..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: pyproject.toml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo -e "${BLUE}📦 Step 1: Installing UV...${NC}"
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ UV already installed${NC}"
else
    pip install uv
    echo -e "${GREEN}✓ UV installed${NC}"
fi

echo -e "${BLUE}🐍 Step 2: Creating virtual environment...${NC}"
uv venv .venv --python 3.11
echo -e "${GREEN}✓ Virtual environment created at .venv${NC}"

echo -e "${BLUE}📥 Step 3: Installing dependencies...${NC}"
uv pip install -e ".[dev,docker]"
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo -e "${BLUE}🔒 Step 4: Creating lock file...${NC}"
uv pip compile pyproject.toml -o requirements.lock --upgrade 2>/dev/null || echo -e "${YELLOW}⚠ Could not create lock file (optional)${NC}"

echo -e "${BLUE}📝 Step 5: Creating .env file...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env from .env.example${NC}"
    echo -e "${YELLOW}⚠ Please edit .env and add your API keys${NC}"
else
    echo -e "${GREEN}✓ .env already exists${NC}"
fi

echo -e "${BLUE}📁 Step 6: Creating log directory...${NC}"
mkdir -p logs
echo -e "${GREEN}✓ Log directory ready${NC}"

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate  # Linux/Mac"
echo "  .venv\\Scripts\\activate     # Windows"
echo ""
echo "To run tests:"
echo "  pytest backend/tests/ -v"
echo ""
echo "To run linting:"
echo "  ruff check backend/"
echo "  ruff format backend/"
echo "  mypy backend/rlm"
echo ""
echo -e "${YELLOW}⚠ Remember to add your API keys to .env file!${NC}"
