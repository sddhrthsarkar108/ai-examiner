# Automated Answer Script Checking System

An intelligent system that uses OCR and AI to evaluate handwritten answer scripts, providing automated grading and feedback.

## Features

- **OCR Processing**: Extracts handwritten text from PDF answer scripts using GPT-4 Vision
- **AI Evaluation**: Evaluates answers based on clarity, completeness, and accuracy
- **Structured Output**: Generates detailed evaluation reports in CSV format
- **Support for Diagrams**: Detects and processes diagrams in answers
- **Flexible Question Types**: Handles both short and long-form questions

## Prerequisites

- Python 3.x
- OpenAI API key
- DeepSeek API key
- Poppler (for PDF processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Maverick-81-cpu/automated_answer_script.git
cd automated_answer_script
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## Usage

1. Run the script:
```bash
python automated_ans_script_checking_v4.py
```

2. When prompted, provide the path to the answer script PDF

3. The system will:
   - Extract text from the PDF
   - Evaluate the answers
   - Generate a CSV report

## Output

The system generates:
- `evaluation_summary.csv`: Contains detailed evaluation results
- `extracted_answers.pdf`: Processed PDF with extracted text

## Project Structure

- `automated_ans_script_checking_v4.py`: Main script for answer evaluation
- `TrOCR_script_v1.py`: Text extraction utilities
- Various versions of scripts showing project evolution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Saswati Sarkar

## Acknowledgments

- OpenAI for GPT-4 Vision API
- DeepSeek for evaluation API
- All contributors and users of this project 