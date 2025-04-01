#!/usr/bin/env bash

# ANSI color codes for prettier output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}==== Environment Verification Tool ====${NC}"
echo -e "${BLUE}This script checks if your environment is correctly set up for the Automated Answer Script Grading project.${NC}"
echo ""

# Check Python version
echo -e "${BOLD}Checking Python:${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python is installed: $PYTHON_VERSION${NC}"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓ Python is installed: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Python 3.8 or newer.${NC}"
fi

# Check pip
echo -e "\n${BOLD}Checking pip:${NC}"
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
    echo -e "${GREEN}✓ pip is installed (pip3)${NC}"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
    echo -e "${GREEN}✓ pip is installed${NC}"
else
    echo -e "${RED}✗ pip is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install pip.${NC}"
fi

# Check direnv
echo -e "\n${BOLD}Checking direnv:${NC}"
if command -v direnv &> /dev/null; then
    echo -e "${GREEN}✓ direnv is installed${NC}"
    
    # Check if .envrc is allowed
    if direnv status 2>/dev/null | grep -q "Found RC allowed true"; then
        echo -e "${GREEN}✓ .envrc is allowed${NC}"
    else
        echo -e "${YELLOW}⚠ .envrc is not allowed${NC}"
        echo -e "${YELLOW}Run 'direnv allow' in the project directory to allow it.${NC}"
    fi
else
    echo -e "${YELLOW}⚠ direnv is not installed${NC}"
    echo -e "${YELLOW}Environment variables will not be automatically loaded.${NC}"
    echo -e "${YELLOW}Either install direnv or manually source the .envrc.local file.${NC}"
fi

# Check asdf
echo -e "\n${BOLD}Checking asdf:${NC}"
if command -v asdf &> /dev/null; then
    echo -e "${GREEN}✓ asdf is installed${NC}"
    
    # Check Python plugin
    if asdf plugin list | grep -q "python"; then
        echo -e "${GREEN}✓ asdf python plugin is installed${NC}"
    else
        echo -e "${YELLOW}⚠ asdf python plugin is not installed${NC}"
        echo -e "${YELLOW}Run 'asdf plugin add python' to install it.${NC}"
    fi
    
    # Check direnv plugin
    if asdf plugin list | grep -q "direnv"; then
        echo -e "${GREEN}✓ asdf direnv plugin is installed${NC}"
    else
        echo -e "${YELLOW}⚠ asdf direnv plugin is not installed${NC}"
        echo -e "${YELLOW}Run 'asdf plugin add direnv' to install it.${NC}"
    fi
else
    echo -e "${YELLOW}⚠ asdf is not installed${NC}"
    echo -e "${YELLOW}Using system Python and direnv.${NC}"
    echo -e "${YELLOW}This is fine if you've set up your environment manually.${NC}"
fi

# Check poppler
echo -e "\n${BOLD}Checking poppler:${NC}"
if command -v pdftoppm &> /dev/null; then
    echo -e "${GREEN}✓ poppler is installed${NC}"
else
    echo -e "${RED}✗ poppler is not installed or not in PATH${NC}"
    echo -e "${YELLOW}On macOS: brew install poppler${NC}"
    echo -e "${YELLOW}On Ubuntu/Debian: apt-get install poppler-utils${NC}"
fi

# Check Python packages
echo -e "\n${BOLD}Checking required Python packages:${NC}"
REQUIRED_PACKAGES=(
    "pdf2image"
    "pandas"
    "numpy"
    "langchain"
    "langchain_openai"
    "langchain_deepseek"
    "langchain_google_genai"
    "dotenv"
    "PIL"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if $PYTHON_CMD -c "import $package" &> /dev/null; then
        echo -e "${GREEN}✓ $package is installed${NC}"
    else
        echo -e "${RED}✗ $package is not installed${NC}"
    fi
done

# Check API keys
echo -e "\n${BOLD}Checking API keys:${NC}"
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file exists${NC}"
    
    # Source .env file
    source .env
    
    # Check OpenAI API key
    if [ -n "$OPENAI_API_KEY" ]; then
        echo -e "${GREEN}✓ OPENAI_API_KEY is set${NC}"
    else
        echo -e "${YELLOW}⚠ OPENAI_API_KEY is not set in .env${NC}"
    fi
    
    # Check Google API key
    if [ -n "$GOOGLE_API_KEY" ]; then
        echo -e "${GREEN}✓ GOOGLE_API_KEY is set${NC}"
    else
        echo -e "${YELLOW}⚠ GOOGLE_API_KEY is not set in .env${NC}"
    fi
    
    # Check DeepSeek API key
    if [ -n "$DEEPSEEK_API_KEY" ]; then
        echo -e "${GREEN}✓ DEEPSEEK_API_KEY is set${NC}"
    else
        echo -e "${YELLOW}⚠ DEEPSEEK_API_KEY is not set in .env${NC}"
    fi
else
    echo -e "${RED}✗ .env file does not exist${NC}"
    echo -e "${YELLOW}Create a .env file with your API keys:${NC}"
    echo -e "${YELLOW}OPENAI_API_KEY=your_key_here${NC}"
    echo -e "${YELLOW}GOOGLE_API_KEY=your_key_here${NC}"
    echo -e "${YELLOW}DEEPSEEK_API_KEY=your_key_here${NC}"
fi

echo -e "\n${BLUE}==== Verification Complete ====${NC}"
echo -e "${BLUE}If all checks passed, your environment is ready to use.${NC}"
echo -e "${BLUE}If some checks failed, follow the instructions to fix them.${NC}"
echo -e "${BLUE}You can run this script again after making changes.${NC}" 