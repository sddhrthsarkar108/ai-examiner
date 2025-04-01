# Automated Answer Script Grading

This project automatically grades scanned handwritten answer scripts for exams using OCR (Optical Character Recognition) and AI evaluation with LangChain.

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
- **LangChain integration for model abstraction and interchangeability**
- **Support for multiple model providers** (OpenAI, DeepSeek)

## Setup

### Requirements

- Python 3.8+
- Poppler (for PDF processing)
- OpenAI API key
- DeepSeek API key

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Poppler:
   - **macOS**: `brew install poppler`
   - **Ubuntu/Debian**: `apt-get install poppler-utils`
   - **Windows**: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases/)

4. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

### Configuration

The application uses several configuration files:

1. `config/app_config.json`: General application settings
2. `config/questions.json`: Exam questions and marks

Default configurations will be created on first run if they don't exist.

#### Configuration Options

##### OCR Configuration Options

The OCR settings control how text is extracted from images:

- `max_tokens`: Maximum number of tokens for the OCR model response (default: 4000)
- `temperature`: Controls randomness in the OCR model (default: 0.0)
  - Lower values (0.0) produce more consistent extraction results
  - Higher values introduce more variation
- `include_examples`: Whether to include examples in the OCR prompt
- `detail_level`: Level of detail for extraction ("high", "medium", "low")
- `model`: The model to use for OCR (default: "gpt-4-vision-preview")
- `provider`: The AI provider to use (default: "openai")

##### Evaluation Configuration Options

The evaluation settings control how answers are graded:

- `num_evaluations`: Number of evaluation runs to perform (default: 3)
- `max_tokens`: Maximum number of tokens for the evaluation model response (default: 1500)
- `temperature`: Controls randomness in the evaluation model (default: 0.1)
  - Lower values produce more consistent grading
  - Slightly higher values (0.1) allow for more flexibility in evaluation
- `model`: The model to use for evaluation (default: "deepseek-chat")
- `provider`: The AI provider to use (default: "deepseek")

##### General Configuration Options

- `domain_context`: Subject area context (e.g., "machine learning exam")
- `output_directory`: Directory for storing results
- `log_level`: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR")

## LangChain Integration

This project uses LangChain to abstract the interaction with different LLM providers:

- **Model Abstraction**: The code interacts with a consistent API regardless of the underlying model
- **Easy Model Swapping**: Change models by updating the configuration without code changes
- **Provider Flexibility**: Support for OpenAI, DeepSeek, and ability to add more providers
- **Maintainable Architecture**: Decoupled from specific provider implementation details

### Currently Supported Providers:

- **OpenAI**: Used for OCR with vision capabilities (default for OCR)
- **DeepSeek**: Used for evaluation (default for evaluation)

## LLM Factory Integration

This project uses a centralized `LLMFactory` to abstract the interaction with different LLM providers:

- **Model Provider Abstraction**: The factory provides a unified interface to OpenAI, Google, and DeepSeek models
- **Default Provider Selection**: 
  - OCR: OpenAI (default: gpt-4o) - best for vision/image processing
  - Interpreter: Google (default: gemini-2.5-pro-exp-03-25) - excellent for structuring and organizing content
  - Evaluation: DeepSeek (default: deepseek-chat) - good balance of accuracy and cost-effectiveness

### Configurable Model Selection

You can change the model provider for any component by updating the `provider` field in the app_config.json:

```json
{
  "ocr": {
    "model": "gpt-4o",
    "provider": "openai"
  },
  "interpreter": {
    "model": "gemini-2.5-pro-exp-03-25",
    "provider": "google"
  },
  "evaluation": {
    "model": "deepseek-chat",
    "provider": "deepseek"
  }
}
```

The system will automatically use the appropriate API key from your environment variables based on the selected provider.

## Centralized Prompts

This project uses a centralized prompts module (`prompts.py`) to manage all system and user prompts used throughout the application. This approach provides several benefits:

- **Better Maintainability**: All prompts are defined in one place, making them easier to update and maintain
- **Consistent Formatting**: Ensures consistent prompt structure across the application
- **Clear Boundaries**: Reinforces the responsibility boundaries between components
- **Easier Experimentation**: Facilitates experimenting with different prompt styles without changing core logic

### Prompt Functions

The `prompts.py` module provides the following functions:

- **OCR Prompts**
  - `get_ocr_system_prompt(domain_context)`: System prompt for OCR text extraction
  - `get_ocr_user_prompt(domain_context, include_examples)`: User prompt for OCR with optional examples

- **Interpreter Prompts**
  - `get_interpreter_system_prompt()`: System prompt for interpreting OCR output
  - `get_interpreter_user_prompt(all_ocr_content)`: User prompt for the interpreter with OCR content

- **Evaluation Prompts**
  - `get_evaluation_system_prompt(domain_context)`: System prompt for evaluation
  - `get_evaluation_user_prompt(domain_context, questions_table, answers, question_score_template)`: User prompt for evaluation

### Domain-Specific Examples

The module also includes a collection of domain-specific examples for machine learning content, which can be used to provide few-shot learning examples to the OCR model for better recognition of mathematical notation, diagrams, and calculations.

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

## Usage

1. Run the application:
   ```