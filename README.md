# Automated Answer Script Grading

This project automatically grades scanned handwritten answer scripts for exams using OCR (Optical Character Recognition) and AI evaluation.

## Project Structure

The codebase is organized into the following modules:

- `main.py`: Main entry point for the application
- `config.py`: Handles configuration loading and management
- `ocr.py`: OCR-related functionality for extracting text from images
- `evaluation.py`: Evaluates and grades extracted answers
- `utils.py`: Utility functions used across the application

## Features

- Extracts handwritten text from scanned PDF answer sheets
- Evaluates answers based on predefined criteria
- Supports multiple evaluation runs for consistent grading
- Uses voting-based system to determine final scores
- Generates detailed scoring reports and statistics
- Configurable domain context for different subjects
- Adjustable AI parameters for OCR and evaluation
- **Support for multiple model providers** (OpenAI, Google, DeepSeek)

## Setup

### Requirements

- Python 3.8+
- Poppler (for PDF processing)
- API keys for supported model providers (OpenAI, Google, DeepSeek)

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
   or
   ```
   pip install -e .
   ```

3. Install Poppler:
   - **macOS**: `brew install poppler`
   - **Ubuntu/Debian**: `apt-get install poppler-utils`
   - **Windows**: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases/)

4. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

### Environment Setup

The project uses `direnv` to manage environment variables and Python versions. There are two ways to set up your environment:

#### Option 1: Using asdf and direnv (Recommended)

This approach provides version-controlled Python environments:

1. Install `asdf` (version manager):
   ```bash
   # On macOS with Homebrew
   brew install asdf

   # Add to your shell profile (.zshrc or .bash_profile)
   echo -e "\n. $(brew --prefix asdf)/libexec/asdf.sh" >> ~/.zshrc
   ```

2. Install required tools automatically using the `.tool-versions` file:
   ```bash
   # Navigate to the project directory
   cd /path/to/project
   
   # Install all tools specified in .tool-versions
   asdf install
   
   # This will automatically install:
   # - direnv 2.35.0
   # - python 3.12.0
   ```

   Alternatively, you can install tools individually:
   ```bash
   asdf plugin add direnv
   asdf install direnv 2.35.0
   asdf global direnv 2.35.0
   
   asdf plugin add python
   asdf install python 3.12.0
   ```

3. Configure direnv hook in your shell:
   ```bash
   # Add direnv hook to your shell
   echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc  # for zsh
   # or
   echo 'eval "$(direnv hook bash)"' >> ~/.bash_profile  # for bash
   ```

4. Allow the `.envrc` file:
   ```bash
   cd /path/to/project
   direnv allow
   ```

#### Option 2: Manual Setup (Without asdf)

If you prefer not to use asdf, you can set up the environment manually:

1. Install `direnv` directly:
   ```bash
   # On macOS
   brew install direnv
   
   # Add direnv hook to your shell
   echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc  # for zsh
   # or
   echo 'eval "$(direnv hook bash)"' >> ~/.bash_profile  # for bash
   ```

2. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Use the `.envrc` file:
   ```bash
   # The repository includes a ready-to-use template
   # Allow it with direnv
   direnv allow .envrc
   ```

   Or create your own `.envrc` file without asdf dependencies:
   ```bash
   # Edit .envrc to remove 'use asdf' line
   ```

4. Alternatively, you can simply run the script and let it fall back to system Python:
   ```bash
   # The script will detect missing direnv and use system Python
   ./run_grading.sh
   ```

> **Note**: The `run_grading.sh` script is designed with robust fallback mechanisms. Even if you don't have asdf or direnv set up, it will attempt to use your system Python and check for all required dependencies, installing them if necessary. This makes it possible to run the application with minimal setup.

### Verify Your Environment

The project includes a verification script to help you check if your environment is correctly set up:

```bash
# Make the script executable if it isn't already
chmod +x verify_env.sh

# Run the verification script
./verify_env.sh
```

This script checks for:
- Python and pip installation
- direnv and asdf setup (if applicable)
- Required dependencies (poppler, Python packages)
- Existence and content of API keys in `.env`

The script provides clear guidance on what's properly configured and what needs attention, helping you troubleshoot any setup issues before running the main application.

### Configuration

The application uses a flexible configuration system with sensible defaults, environment variables, and optional configuration files.

#### Configuration Options

##### OCR Configuration Options

The OCR settings control how text is extracted from images:

- `max_tokens`: Maximum number of tokens for the OCR model response (default: 4000)
- `temperature`: Controls randomness in the OCR model (default: 0.0)
  - Lower values (0.0) produce more consistent extraction results
- `include_examples`: Whether to include examples in the OCR prompt
- `detail_level`: Level of detail for extraction ("high", "medium", "low")
- `model`: The model to use for OCR (default: "gpt-4o")
- `provider`: The AI provider to use (default: "openai")

##### Evaluation Configuration Options

The evaluation settings control how answers are graded:

- `num_evaluations`: Number of evaluation runs to perform (default: 3)
- `max_tokens`: Maximum number of tokens for the evaluation model response (default: 1500)
- `temperature`: Controls randomness in the evaluation model (default: 0.1)
  - Lower values produce more consistent grading
- `model`: The model to use for evaluation (default: "deepseek-chat")
- `provider`: The AI provider to use (default: "deepseek")

##### General Configuration Options

- `domain_context`: Subject area context (e.g., "machine learning exam")
- `output_directory`: Directory for storing results
- `log_level`: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR")

## Improved Configuration Management

The application uses a robust environment-based configuration system that allows for single-trigger execution:

- **Environment Variables**: All configurations are loaded from environment variables, leveraging direnv for automatic loading.
- **No Interactive Prompts**: The application can run without any user interaction when properly configured.
- **Default Values**: Sensible defaults are applied for any missing environment variables.
- **Config Display**: Optional configuration display can be enabled in the environment.

### Environment Configuration

The application uses `.envrc` and `.env` files to manage configurations:

1. **`.env`**: Contains sensitive API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

2. **`.envrc`**: Contains all application settings and sources the `.env` file:
   ```bash
   # Source API keys
   source_env .env

   # OCR Configuration
   export OCR_MAX_TOKENS=3000
   export OCR_INCLUDE_EXAMPLES=true
   export OCR_DETAIL_LEVEL=high
   export OCR_TEMPERATURE=0.0
   export OCR_MODEL=gpt-4o
   export OCR_PROVIDER=openai

   # Interpreter Configuration
   export INTERPRETER_MAX_TOKENS=4000
   export INTERPRETER_TEMPERATURE=0.1
   export INTERPRETER_MODEL=gemini-2.5-pro
   export INTERPRETER_PROVIDER=google

   # Evaluation Configuration
   export EVALUATION_NUM_EVALUATIONS=3
   export EVALUATION_MAX_TOKENS=3000
   export EVALUATION_TEMPERATURE=0.1
   export EVALUATION_MODEL=deepseek-chat
   export EVALUATION_PROVIDER=deepseek

   # General Configuration
   export GENERAL_DOMAIN_CONTEXT="machine learning exam"
   export GENERAL_OUTPUT_DIRECTORY="output"
   export GENERAL_LOG_LEVEL="INFO"

   # App Behavior Settings
   export SHOW_CONFIG=false
   export DEFAULT_PDF_PATH="/path/to/default/exam.pdf"
   export AUTO_REUSE_OCR=true
   export AUTO_REUSE_INTERPRETER=true
   ```

### Behavior Settings

Special environment variables control the application's behavior for non-interactive use:

| Variable | Purpose | Values |
|----------|---------|--------|
| `SHOW_CONFIG` | Whether to display config on startup | `true` or `false` |
| `DEFAULT_PDF_PATH` | Default PDF to process | File path or empty |
| `AUTO_REUSE_OCR` | Auto-reuse existing OCR results | `true` or `false` |
| `AUTO_REUSE_INTERPRETER` | Auto-reuse existing interpreter results | `true` or `false` |
| `PROCESS_ALL_FILES` | Process all files in directory | `true` or `false` |
| `SKIP_EXISTING_OUTPUTS` | Skip files with existing output | `true` or `false` |

With these settings, you can configure the application to run completely non-interactively:

```bash
# Set up for complete automation
export DEFAULT_PDF_PATH="/path/to/exam.pdf"
export AUTO_REUSE_OCR=true
export AUTO_REUSE_INTERPRETER=true
```

### API Key Management

The application provides intelligent defaults for API keys with a flexible priority system:

1. **Environment Variables**: The system checks for provider-specific environment variables.
2. **Default Keys**: Once a key is successfully used, it's cached for future use within the session.

The application looks for these environment variables by default:
- OpenAI: `OPENAI_API_KEY`
- Google: `GOOGLE_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY` or `DEEPINFRA_API_TOKEN`

### Default Provider Selection

The application uses optimized defaults for each processing stage:

- **OCR**: OpenAI (default: gpt-4o) - best for vision/image processing
- **Interpreter**: Google (default: gemini-2.5-pro) - excellent for structuring content
- **Evaluation**: DeepSeek (default: deepseek-chat) - good balance of accuracy and cost

## Usage

### Running the Grading Script

The application includes a `run_grading.sh` script that simplifies the execution process. Here are common usage examples:

#### Processing PDF Files

```bash
# Process a single PDF file
./run_grading.sh answer_sheets/exam01.pdf

# Process a PDF with forced OCR and interpretation (ignores existing results)
./run_grading.sh -f -i answer_sheets/08.pdf

# Process all PDFs in the answer_sheets directory
./run_grading.sh -a

# Process PDFs from a custom directory
./run_grading.sh -d /path/to/custom/pdf/directory
```

#### Non-OCR Mode (Using Pre-extracted Text)

```bash
# Process a specific text file in non-OCR mode
./run_grading.sh -n -t extracted_texts/01_combined.txt

# Process all text files in the extracted_texts directory
./run_grading.sh -n

# Process all text files from a custom directory
./run_grading.sh -n -x /path/to/extracted/texts
```

#### Result Aggregation

```bash
# Process all PDFs and aggregate results
./run_grading.sh -a -g

# Process all text files in non-OCR mode and aggregate results
./run_grading.sh -n -g

# Run aggregation separately after processing
python aggregate_results.py
```

#### Additional Options

```bash
# Show configuration details before running
./run_grading.sh -s answer_sheets/exam01.pdf

# Use multi-provider LLM evaluation (uses different models)
./run_grading.sh -m answer_sheets/exam01.pdf

# Display help with all available options
./run_grading.sh -h
```

For the most up-to-date information, you can always check the help documentation:
```bash
./run_grading.sh -h
```

## Advanced Usage

### Non-OCR Mode

If you already have extracted text files and want to skip the OCR process, you can use the non-OCR mode:

```bash
# Process a specific text file in non-OCR mode
./run_grading.sh -n -t extracted_texts/02_combined.txt

# Process all text files in the extracted_texts directory
./run_grading.sh -n -a
```

Text files should follow the format with page markers (e.g., "=== Page 1 ===") to correctly separate content by page.

### Result Aggregation

After processing multiple files, you can aggregate the results into a master spreadsheet:

```bash
# Process all files and run aggregation after completion
./run_grading.sh -n -a -g

# Run aggregation separately
python aggregate_results.py
```

The aggregation script combines all individual evaluation summaries into a master spreadsheet, making it easier to compare results across different files.
