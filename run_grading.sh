#!/usr/bin/env bash
set -e

# ANSI color codes for prettier output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check and allow direnv at the beginning
if command -v direnv &> /dev/null; then
  # Check if .envrc file exists in the current directory
  if [ -f ".envrc" ]; then
    # Check if .envrc file is blocked
    if direnv status 2>/dev/null | grep -q "Blocked"; then
      echo -e "${YELLOW}The .envrc file is blocked. Running 'direnv allow' to approve its content...${NC}"
      direnv allow
      if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully allowed .envrc file${NC}"
        # Reload direnv to apply settings immediately
        eval "$(direnv export bash)"
        echo -e "${GREEN}Direnv configuration loaded${NC}"
      else
        echo -e "${RED}Failed to allow .envrc file. Please run 'direnv allow' manually.${NC}"
        exit 1
      fi
    else
      # Load environment variables from .envrc
      eval "$(direnv export bash)"
      echo -e "${GREEN}Direnv configuration loaded${NC}"
    fi
  else
    echo -e "${YELLOW}Warning: No .envrc file found in the current directory.${NC}"
  fi
else
  echo -e "${YELLOW}Warning: direnv not found. Configuration from .envrc will not be loaded.${NC}"
  echo "Please install direnv or manually source the .envrc file."
fi

# Sections for help message
NON_OCR_HELP="
${BOLD}Non-OCR Mode:${NC}
  Use -n/--non-ocr to process text files directly instead of PDFs.
  Specify a text file with -t/--text-file or process all files with -a/--all.
  Text files should be placed in the extracted_texts/ directory by default.
  You can change this directory with -x/--extracted-dir.
"

AGGREGATION_HELP="
${BOLD}Result Aggregation:${NC}
  Use -g/--aggregate to combine all individual results into a master spreadsheet.
  The script will:
    1. Find the latest run folder for each processed file
    2. Extract data from the aggregate_evaluation_summary.csv
    3. Combine all data into a master Excel spreadsheet in the output directory
  
  You can also run the aggregation script separately:
    python aggregate_results.py
"

# Helper function for displaying usage information
usage() {
  echo -e "${BOLD}Usage:${NC} $0 [options] [PDF_PATH]"
  echo ""
  echo -e "${BOLD}Options:${NC}"
  echo "  -h, --help                  Show this help message"
  echo "  -s, --show-config           Show configuration details before running"
  echo "  -f, --force-ocr             Force re-running OCR even if results exist"
  echo "  -i, --force-interpreter     Force re-running interpreter even if results exist"
  echo "  -a, --all                   Process all PDFs in answer_sheets/ directory"
  echo "  -d, --dir DIR               Process all PDFs in specified directory"
  echo "  -m, --multi-llm             Enable multi-provider LLM evaluation (uses different models)"
  echo "  -n, --non-ocr               Enable non-OCR mode to use pre-extracted text files"
  echo "  -t, --text-file FILE        Specify text file to use in non-OCR mode"
  echo "  -x, --extracted-dir DIR     Specify directory containing extracted text files (default: extracted_texts)"
  echo "  -g, --aggregate             Run aggregation script after processing to create master spreadsheet"
  echo ""
  echo -e "${BOLD}Force Processing:${NC}"
  echo "  Use -f to force re-running OCR, or -i to force re-running interpreter."
  echo "  You can also set SKIP_EXISTING_OUTPUTS=false in .envrc to always process files regardless of existing output."
  
  echo -e "$NON_OCR_HELP"
  echo -e "$AGGREGATION_HELP"
  
  echo -e "${BOLD}Example Usage:${NC}"
  echo "  $0 -a"
  echo "  $0 -n -t extracted_texts/1_combined.txt"
  echo "  $0 -d /path/to/pdfs -g"
  exit 1
}

# Function to check for required dependencies
check_dependencies() {
  echo -e "${BLUE}Checking for required dependencies...${NC}"
  
  # Check if direnv is properly loaded and Python is available within that environment
  if [[ -n "$DIRENV_DIR" ]]; then
    echo -e "${GREEN}Using direnv managed Python environment${NC}"
    # Direnv is active, use the Python from .direnv
    if command -v python &> /dev/null; then
      PYTHON_CMD="python"
      PYTHON_VERSION=$(python --version 2>&1)
      echo -e "${GREEN}Using direnv Python: $PYTHON_VERSION${NC}"
    else
      echo -e "${RED}Error: Python not found in direnv environment. Please ensure your .envrc sets up Python correctly.${NC}"
      exit 1
    fi
    
    # Find the correct pip command
    if command -v pip &> /dev/null; then
      PIP_CMD="pip"
    else
      echo -e "${RED}Error: pip not found in direnv environment. Please ensure your .envrc sets up pip correctly.${NC}"
      exit 1
    fi
  else
    # Direnv is not active, fallback to system Python
    echo -e "${YELLOW}Direnv environment not detected, falling back to system Python${NC}"
    
    # Check if Python exists in the current environment
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
      echo -e "${RED}Error: Neither python nor python3 command found. Please ensure Python is installed and in your PATH.${NC}"
      exit 1
    fi
    
    # Find the correct Python command
    if command -v python &> /dev/null; then
      PYTHON_CMD="python"
    else
      PYTHON_CMD="python3"
    fi
    
    # Check if pip exists
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
      echo -e "${RED}Error: Neither pip nor pip3 command found. Please ensure pip is installed.${NC}"
      exit 1
    fi
    
    # Find the correct pip command
    if command -v pip &> /dev/null; then
      PIP_CMD="pip"
    else
      PIP_CMD="pip3"
    fi
  fi
  
  # Display Python environment info
  echo -e "${BLUE}Using Python: $(which $PYTHON_CMD)${NC}"
  echo -e "${BLUE}Python version: $($PYTHON_CMD --version 2>&1)${NC}"
  
  # Check for required Python packages
  REQUIRED_PACKAGES=(
    "pdf2image"
    "pandas"
    "numpy"
    "python-dotenv"
    "langchain"
    "langchain-openai"
    "langchain-deepseek"
    "langchain-google-genai"
    "Pillow"
  )
  
  MISSING_PACKAGES=()
  for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! $PYTHON_CMD -c "import $package" &> /dev/null; then
      if [[ "$package" == "python-dotenv" ]]; then
        # Special case for python-dotenv since it's imported as 'dotenv'
        if ! $PYTHON_CMD -c "import dotenv" &> /dev/null; then
          MISSING_PACKAGES+=("$package")
        fi
      elif [[ "$package" == "langchain" ]]; then
        # Special case for langchain packages
        if ! $PYTHON_CMD -c "import langchain" &> /dev/null; then
          MISSING_PACKAGES+=("$package")
        fi
      elif [[ "$package" == "langchain-openai" ]]; then
        if ! $PYTHON_CMD -c "import langchain_openai" &> /dev/null; then
          MISSING_PACKAGES+=("$package")
        fi
      elif [[ "$package" == "langchain-deepseek" ]]; then
        if ! $PYTHON_CMD -c "import langchain_deepseek" &> /dev/null; then
          MISSING_PACKAGES+=("$package")
        fi
      elif [[ "$package" == "langchain-google-genai" ]]; then
        if ! $PYTHON_CMD -c "import langchain_google_genai" &> /dev/null; then
          MISSING_PACKAGES+=("$package")
        fi
      else
        MISSING_PACKAGES+=("$package")
      fi
    fi
  done
  
  if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}The following required packages are missing: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "Installing required packages..."
    
    # Install missing packages
    $PIP_CMD install "${MISSING_PACKAGES[@]}"
    
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}Successfully installed required packages${NC}"
    else
      echo -e "${RED}Failed to install required packages. Please install them manually:${NC}"
      echo -e "$PIP_CMD install ${MISSING_PACKAGES[*]}"
      exit 1
    fi
  else
    echo -e "${GREEN}All required Python packages are installed.${NC}"
  fi
  
  # Check for Poppler dependency (required for pdf2image)
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check if poppler is installed via Homebrew
    if ! command -v pdftoppm &> /dev/null; then
      echo -e "${YELLOW}Poppler is required but not found. Attempting to install via Homebrew...${NC}"
      if command -v brew &> /dev/null; then
        brew install poppler
      else
        echo -e "${RED}Homebrew not found. Please install poppler manually or install Homebrew first.${NC}"
        echo -e "Visit: https://brew.sh/"
        exit 1
      fi
    fi
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - check for poppler-utils
    if ! command -v pdftoppm &> /dev/null; then
      echo -e "${YELLOW}Poppler is required but not found. Attempting to install poppler-utils...${NC}"
      if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update && sudo apt-get install -y poppler-utils
      elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install poppler-utils
      else
        echo -e "${RED}Could not install poppler-utils. Please install manually.${NC}"
        exit 1
      fi
    fi
  fi
  
  echo -e "${GREEN}All dependencies are satisfied.${NC}"
}

# Parse command line arguments
PDF_PATH=""
SHOW_CONFIG="false"
FORCE_OCR="false"
FORCE_INTERPRETER="false"
PROCESS_ALL="false"
CUSTOM_DIR=""
MULTI_LLM="false"
NON_OCR_MODE="false"
TEXT_FILE=""
EXTRACTED_DIR=""
RUN_AGGREGATION="true"

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      ;;
    -s|--show-config)
      SHOW_CONFIG="true"
      shift
      ;;
    -f|--force-ocr)
      FORCE_OCR="true"
      shift
      ;;
    -i|--force-interpreter)
      FORCE_INTERPRETER="true"
      shift
      ;;
    -a|--all)
      PROCESS_ALL="true"
      shift
      ;;
    -d|--dir)
      CUSTOM_DIR="$2"
      shift 2
      ;;
    -m|--multi-llm)
      MULTI_LLM="true"
      shift
      ;;
    -n|--non-ocr)
      NON_OCR_MODE="true"
      shift
      ;;
    -t|--text-file)
      TEXT_FILE="$2"
      shift 2
      ;;
    -x|--extracted-dir)
      EXTRACTED_DIR="$2"
      shift 2
      ;;
    -g|--aggregate)
      RUN_AGGREGATION="true"
      shift
      ;;
    *)
      # Assume it's the PDF path or text file path
      PDF_PATH="$1"
      shift
      ;;
  esac
done

# Check dependencies
check_dependencies

# Create answer_sheets directory if it doesn't exist
if [ ! -d "answer_sheets" ]; then
  mkdir -p "answer_sheets"
  echo -e "${BLUE}Created 'answer_sheets' directory${NC}"
fi

# Create extracted_texts directory if it doesn't exist
if [ ! -d "extracted_texts" ]; then
  mkdir -p "extracted_texts"
  echo -e "${BLUE}Created 'extracted_texts' directory${NC}"
fi

# Function to process a single PDF file
process_pdf() {
  local pdf_file="$1"
  
  if [ ! -f "$pdf_file" ]; then
    echo -e "${RED}Error: PDF file not found: $pdf_file${NC}"
    return 1
  fi
  
  echo -e "\n${BLUE}==================================================================${NC}"
  echo -e "${GREEN}Processing: ${BOLD}$(basename "$pdf_file")${NC}"
  echo -e "${BLUE}==================================================================${NC}"
  
  # Set environment variables for this run
  export DEFAULT_PDF_PATH="$pdf_file"
  export SHOW_CONFIG="$SHOW_CONFIG"
  export AUTO_REUSE_OCR="$([ "$FORCE_OCR" == "true" ] && echo "false" || echo "true")"
  export AUTO_REUSE_INTERPRETER="$([ "$FORCE_INTERPRETER" == "true" ] && echo "false" || echo "true")"
  export OCR_NON_OCR_MODE="false"  # Ensure we're in normal OCR mode
  
  # Set multi-LLM evaluation if enabled
  if [ "$MULTI_LLM" == "true" ]; then
    echo -e "${YELLOW}Multi-provider LLM evaluation enabled${NC}"
    echo -e "${YELLOW}Using DeepSeek for runs 1 & 3, and Google/Gemini for run 2${NC}"
    export EVALUATION_MULTI_PROVIDER_EVALUATION="true"
  else
    export EVALUATION_MULTI_PROVIDER_EVALUATION="false"
  fi
  
  # Run the Python script
  $PYTHON_CMD main.py
  
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo -e "${RED}Error processing ${pdf_file}. Exit code: $exit_code${NC}"
    return $exit_code
  fi
  
  echo -e "${GREEN}Successfully processed: $(basename "$pdf_file")${NC}"
  echo -e "${BLUE}==================================================================${NC}\n"
  return 0
}

# Function to process a single text file in non-OCR mode
process_text_file() {
  local text_file="$1"
  
  if [ ! -f "$text_file" ]; then
    echo -e "${RED}Error: Text file not found: $text_file${NC}"
    return 1
  fi
  
  echo -e "\n${BLUE}==================================================================${NC}"
  echo -e "${GREEN}Processing in NON-OCR mode: ${BOLD}$(basename "$text_file")${NC}"
  echo -e "${BLUE}==================================================================${NC}"
  
  # Set environment variables for this run
  export DEFAULT_PDF_PATH="$text_file"
  export SHOW_CONFIG="$SHOW_CONFIG"
  export AUTO_REUSE_OCR="$([ "$FORCE_OCR" == "true" ] && echo "false" || echo "true")"
  export AUTO_REUSE_INTERPRETER="$([ "$FORCE_INTERPRETER" == "true" ] && echo "false" || echo "true")"
  export OCR_NON_OCR_MODE="true"  # Enable non-OCR mode
  
  # Set custom extracted_texts directory if provided
  if [ -n "$EXTRACTED_DIR" ]; then
    export OCR_EXTRACTED_TEXTS_DIR="$EXTRACTED_DIR"
    echo -e "${BLUE}Using custom extracted texts directory: $EXTRACTED_DIR${NC}"
  fi
  
  # Set multi-LLM evaluation if enabled
  if [ "$MULTI_LLM" == "true" ]; then
    echo -e "${YELLOW}Multi-provider LLM evaluation enabled${NC}"
    echo -e "${YELLOW}Using DeepSeek for runs 1 & 3, and Google/Gemini for run 2${NC}"
    export EVALUATION_MULTI_PROVIDER_EVALUATION="true"
  else
    export EVALUATION_MULTI_PROVIDER_EVALUATION="false"
  fi
  
  # Run the Python script
  $PYTHON_CMD main.py
  
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo -e "${RED}Error processing ${text_file}. Exit code: $exit_code${NC}"
    return $exit_code
  fi
  
  echo -e "${GREEN}Successfully processed: $(basename "$text_file")${NC}"
  echo -e "${BLUE}==================================================================${NC}\n"
  return 0
}

# Process files based on command line arguments
if [ "$NON_OCR_MODE" == "true" ]; then
  # Non-OCR mode: Process text files instead of PDFs
  echo -e "${YELLOW}Running in NON-OCR mode. Will use pre-extracted text files.${NC}"
  
  if [ -n "$TEXT_FILE" ]; then
    # Process the specified text file
    process_text_file "$TEXT_FILE"
  elif [ -n "$PDF_PATH" ]; then
    # PDF_PATH is actually a text file path in non-OCR mode
    process_text_file "$PDF_PATH"
  elif [ "$PROCESS_ALL" == "true" ]; then
    # Process all text files in extracted_texts directory
    TARGET_DIR="${EXTRACTED_DIR:-extracted_texts}"
    echo -e "${BLUE}Processing all text files in ${TARGET_DIR}/ directory...${NC}"
    
    # Find all text files in the directory
    TEXT_FILES=($(find "$TARGET_DIR" -name "*.txt" -type f | sort))
    
    if [ ${#TEXT_FILES[@]} -eq 0 ]; then
      echo -e "${YELLOW}No text files found in ${TARGET_DIR}/ directory.${NC}"
      exit 0
    fi
    
    echo -e "${BLUE}Found ${#TEXT_FILES[@]} text files to process.${NC}"
    
    # Process each text file
    for text_file in "${TEXT_FILES[@]}"; do
      process_text_file "$text_file"
    done
  else
    # Let the Python script handle the input
    echo -e "${BLUE}No text file specified, running with default settings...${NC}"
    
    # Set environment variables
    export SHOW_CONFIG="$SHOW_CONFIG"
    export AUTO_REUSE_OCR="$([ "$FORCE_OCR" == "true" ] && echo "false" || echo "true")"
    export AUTO_REUSE_INTERPRETER="$([ "$FORCE_INTERPRETER" == "true" ] && echo "false" || echo "true")"
    export OCR_NON_OCR_MODE="true"  # Enable non-OCR mode
    export PROCESS_ALL_FILES="true"  # Process all files in extracted_texts by default
    
    # Set custom extracted_texts directory if provided
    if [ -n "$EXTRACTED_DIR" ]; then
      export OCR_EXTRACTED_TEXTS_DIR="$EXTRACTED_DIR"
    fi
    
    # Set multi-LLM evaluation if enabled
    if [ "$MULTI_LLM" == "true" ]; then
      echo -e "${YELLOW}Multi-provider LLM evaluation enabled${NC}"
      echo -e "${YELLOW}Using DeepSeek for runs 1 & 3, and Google/Gemini for run 2${NC}"
      export EVALUATION_MULTI_PROVIDER_EVALUATION="true"
    else
      export EVALUATION_MULTI_PROVIDER_EVALUATION="false"
    fi
    
    # Run the main Python script
    $PYTHON_CMD main.py
  fi
elif [ "$PROCESS_ALL" == "true" ]; then
  # Process all PDFs in answer_sheets directory
  TARGET_DIR="answer_sheets"
  echo -e "${BLUE}Processing all PDFs in ${TARGET_DIR}/ directory...${NC}"
  
  # Find all PDF files in the directory
  PDF_FILES=($(find "$TARGET_DIR" -name "*.pdf" -type f | sort))
  
  if [ ${#PDF_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}No PDF files found in ${TARGET_DIR}/ directory.${NC}"
    exit 0
  fi
  
  echo -e "${BLUE}Found ${#PDF_FILES[@]} PDF files to process.${NC}"
  
  # Process each PDF file
  for pdf_file in "${PDF_FILES[@]}"; do
    process_pdf "$pdf_file"
  done
  
elif [ -n "$CUSTOM_DIR" ]; then
  # Process all PDFs in the specified directory
  if [ ! -d "$CUSTOM_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $CUSTOM_DIR${NC}"
    exit 1
  fi
  
  echo -e "${BLUE}Processing all PDFs in ${CUSTOM_DIR}/ directory...${NC}"
  
  # Find all PDF files in the directory
  PDF_FILES=($(find "$CUSTOM_DIR" -name "*.pdf" -type f | sort))
  
  if [ ${#PDF_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}No PDF files found in ${CUSTOM_DIR}/ directory.${NC}"
    exit 0
  fi
  
  echo -e "${BLUE}Found ${#PDF_FILES[@]} PDF files to process.${NC}"
  
  # Process each PDF file
  for pdf_file in "${PDF_FILES[@]}"; do
    process_pdf "$pdf_file"
  done
  
elif [ -n "$PDF_PATH" ]; then
  # Process the specified PDF file
  process_pdf "$PDF_PATH"
  
else
  # No PDF path provided, use DEFAULT_PDF_PATH from .envrc or prompt
  if [ -n "$DEFAULT_PDF_PATH" ]; then
    echo -e "${BLUE}Using default PDF path: $DEFAULT_PDF_PATH${NC}"
    process_pdf "$DEFAULT_PDF_PATH"
  else
    # Let the Python script handle the input
    echo -e "${BLUE}No PDF path specified, running with default settings...${NC}"
    
    # Set environment variables
    export SHOW_CONFIG="$SHOW_CONFIG"
    export AUTO_REUSE_OCR="$([ "$FORCE_OCR" == "true" ] && echo "false" || echo "true")"
    export AUTO_REUSE_INTERPRETER="$([ "$FORCE_INTERPRETER" == "true" ] && echo "false" || echo "true")"
    
    # Set multi-LLM evaluation if enabled
    if [ "$MULTI_LLM" == "true" ]; then
      echo -e "${YELLOW}Multi-provider LLM evaluation enabled${NC}"
      echo -e "${YELLOW}Using DeepSeek for runs 1 & 3, and Google/Gemini for run 2${NC}"
      export EVALUATION_MULTI_PROVIDER_EVALUATION="true"
    else
      export EVALUATION_MULTI_PROVIDER_EVALUATION="false"
    fi
    
    # Run the main Python script
    $PYTHON_CMD main.py
  fi
fi

echo -e "\n${GREEN}${BOLD}All processing completed!${NC}" 

# Run aggregation script if requested
if [ "$RUN_AGGREGATION" == "true" ]; then
  echo -e "\n${BLUE}Running aggregation script to create master spreadsheet...${NC}"
  $PYTHON_CMD aggregate_results.py
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully created master evaluation spreadsheet${NC}"
  else
    echo -e "${RED}Failed to create master evaluation spreadsheet${NC}"
  fi
fi 