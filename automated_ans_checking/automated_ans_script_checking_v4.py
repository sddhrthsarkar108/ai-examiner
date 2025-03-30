import openai
import base64
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from pdf2image import convert_from_path
import pandas as pd
import re
import requests
from dotenv import load_dotenv
import time
from datetime import datetime
import platform

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set DeepSeek API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("DeepSeek API key not found. Please set DEEPSEEK_API_KEY in your .env file.")
    sys.exit(1)

# Default OCR configuration
OCR_CONFIG = {
    "max_tokens": 4000,  # Reduced from 5000 to 4000 to avoid token limit error
    "include_examples": False,
    "detail_level": "high"
}

# Default evaluation configuration
EVAL_CONFIG = {
    "num_evaluations": 3  # Default number of evaluation runs
}

# Default application configuration
APP_CONFIG = {
    "ocr": OCR_CONFIG.copy(),
    "evaluation": EVAL_CONFIG.copy(),
    "general": {
        "output_directory": "output",
        "log_level": "INFO",
        "domain_context": "machine learning exam"  # Moved domain_context to general section
    }
}

# Load questions from config or use hardcoded default
def load_questions():
    """Load questions from the questions configuration file."""
    questions_config_path = Path(__file__).parent / "config" / "questions.json"
    
    if not questions_config_path.exists():
        logger.error(f"Questions configuration file not found at {questions_config_path}")
        raise FileNotFoundError(f"Questions configuration file not found at {questions_config_path}")
    
    try:
        with open(questions_config_path, 'r') as f:
            questions = json.load(f)
        logger.info(f"Successfully loaded questions from {questions_config_path}")
        return questions
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing questions configuration file: {str(e)}")
        raise json.JSONDecodeError(f"Error parsing questions configuration file: {str(e)}", e.doc, e.pos)
    except Exception as e:
        logger.error(f"Error loading questions configuration: {str(e)}")
        raise

# Get questions
questions = load_questions()

def detect_poppler_path():
    """Detect poppler path based on operating system."""
    system = platform.system()
    if system == "Darwin":  # macOS
        for path in ["/opt/homebrew/bin/", "/usr/local/bin/"]:
            if os.path.exists(path):
                return path
    elif system == "Linux":
        for path in ["/usr/bin/", "/usr/local/bin/"]:
            if os.path.exists(path):
                return path
    elif system == "Windows":
        # Add Windows paths if needed
        return None
    
    logger.warning("Could not detect poppler path. Attempting to use system path.")
    return None

def encode_image_to_base64(image_path):
    """Convert an image file to base64 format."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def extract_text_from_image(image_path, output_dir, app_config=None):
    """Extract handwritten text from an image using OpenAI GPT-4 Vision with enhanced prompt.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save output files
        app_config: Optional application configuration dictionary
    """
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return "Error: Could not encode image."
    
    # Apply custom config if provided, otherwise use default
    config = OCR_CONFIG.copy()
    if app_config and "ocr" in app_config:
        config = app_config["ocr"]
    
    # Get domain_context from general section
    domain_context = "handwritten answer sheet for a machine learning exam"
    if app_config and "general" in app_config:
        domain_context = app_config["general"].get("domain_context", "machine learning exam")
    
    client = openai.OpenAI()
    
    # Save the API call request for audit purposes
    with open(output_dir / "api_request_log.txt", "a") as f:
        f.write(f"\n--- Image OCR Request: {image_path} at {datetime.now().isoformat()} ---\n")
    
    # Create few-shot examples for complex elements if enabled
    examples = ""
    if config["include_examples"]:
        examples = """
Example 1 - Mathematical equation:
[EQUATION] E = mc^2 where m is mass and c is speed of light [/EQUATION]

Example 2 - Figure description:
[FIGURE] A graph showing ROC curve with TPR on y-axis (0-1.0) and FPR on x-axis (0-1.0). The curve shows a concave shape above the diagonal line, with an AUC value of approximately 0.85 written next to it. [/FIGURE]

Example 3 - Calculation steps:
[CALCULATION]
precision = TP/(TP+FP)
= 10/(10+3)
= 10/13
= 0.769
[/CALCULATION]
"""
    
    # Construct domain-specific system prompt
    system_prompt = f"You are an advanced OCR system specialized in extracting handwritten {domain_context} answers. Extract text with high fidelity while maintaining structural integrity and mathematical accuracy."
    
    # Construct detailed user prompt with clear instructions
    user_prompt = f"""Extract all handwritten text from this exam answer sheet.

INSTRUCTIONS:
1. Begin with [ANSWER_START] and end with [ANSWER_END]
2. Use structural markers:
   - Mark questions as [Q#] where # is the question number
   - Mark multi-part answers as [PART a:], [PART b:], etc.

3. For mathematical notation:
   - Enclose equations in [EQUATION]...[/EQUATION] tags
   - Use ^ for superscripts, _ for subscripts
   - Write Greek letters as [alpha], [beta], [theta], etc.
   - Format fractions as (numerator)/(denominator)
   - Preserve matrix structures with [ ] brackets

4. For diagrams/figures:
   - Enclose descriptions in [FIGURE]...[/FIGURE] tags
   - Include what the diagram represents
   - Note any axis labels, legends, or key elements
   - Describe the visual pattern (increasing/decreasing, shape)

5. For calculations:
   - Enclose in [CALCULATION]...[/CALCULATION] tags
   - Show each step on a new line
   - Preserve the progression of the calculation

6. For unclear handwriting:
   - Provide your best guess and mark with [?]
   - If completely illegible, write [ILLEGIBLE]

7. Use formatting:
   - Bold important terms with **term**
   - Create bulleted lists where appropriate
   - Maintain paragraph breaks as in the original

{examples}

Remember that accuracy is critical - this text will be used for automated grading."""
    
    try:
        # Add rate limiting
        time.sleep(1)  # 1 second delay between calls
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=config["max_tokens"]
        )
        extracted_text = response.choices[0].message.content.strip()
        
        # Save extracted text for audit
        with open(output_dir / "extracted_text_log.txt", "a") as f:
            f.write(f"\n--- Extracted from {image_path} at {datetime.now().isoformat()} ---\n")
            f.write(extracted_text)
            f.write("\n---\n")
            
        return extracted_text
    except openai.RateLimitError as e:
        logger.error(f"Rate limit error during OCR extraction: {e}")
        return f"Error: Rate limit exceeded. Please try again later."
    except openai.APIError as e:
        logger.error(f"API error during OCR extraction: {e}")
        return f"Error: API issue. {str(e)}"
    except Exception as e:
        logger.error(f"Error during OCR extraction: {e}")
        return f"Error: {str(e)}"

def extract_answers_from_pdf(pdf_path, output_dir, app_config=None):
    """Convert a multi-page scanned PDF to images and extract handwritten text."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    temp_dir = output_dir / "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        poppler_path = detect_poppler_path()
        conversion_args = {"poppler_path": poppler_path} if poppler_path else {}
        
        logger.info(f"Converting PDF using poppler path: {poppler_path}")
        images = convert_from_path(pdf_path, **conversion_args)
        
        extracted_answers = {}
        for i, image in enumerate(images):
            image_path = temp_dir / f"page_{i + 1}.jpg"
            image.save(image_path, "JPEG")
            
            logger.info(f"Extracting text from page {i + 1}...")
            extracted_text = extract_text_from_image(image_path, output_dir, app_config)
            extracted_answers[f"Page {i + 1}"] = extracted_text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary image files")
    
    # Save extracted answers to file
    with open(output_dir / "extracted_answers.json", "w") as f:
        json.dump(extracted_answers, f, indent=2)
    
    # Process the extracted answers to clean up formatting
    processed_answers = process_extracted_answers(extracted_answers)
    with open(output_dir / "processed_answers.json", "w") as f:
        json.dump(processed_answers, f, indent=2)
    
    return processed_answers

def process_extracted_answers(extracted_answers):
    """Process and clean up extracted answers to make them more consistent."""
    processed = {}
    
    for page, content in extracted_answers.items():
        # Remove the start/end markers if present
        content = re.sub(r'\[ANSWER_START\]|\[ANSWER_END\]', '', content).strip()
        
        # Clean up any double spaces or excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        processed[page] = content
    
    return processed

def evaluate_answers(questions, answers, eval_output_dir, app_config=None):
    """Evaluate answers based on clarity, completeness, accuracy, examples, and presentation.
    
    Args:
        questions: Dictionary of questions and maximum marks
        answers: Student answers to evaluate
        eval_output_dir: Directory to save evaluation results
        app_config: Application configuration dictionary
    """
    # Get domain context from app_config or use default
    domain_context = "machine learning exam"
    if app_config and "general" in app_config:
        domain_context = app_config["general"].get("domain_context", "machine learning exam")
    
    # Construct a table of questions and marks for the prompt
    questions_table = "\n".join([f"Question {q_no}: {question_text} [{max_marks} marks]" 
                               for q_no, (question_text, max_marks) in questions.items()])
    
    prompt = f"""You are evaluating a {domain_context}. Assess these student answers with expert-level knowledge of {domain_context} concepts.

Questions and Marks:
{questions_table}

Student Answers:
{answers}

EVALUATION CRITERIA:
- Correctness (70%): Factual accuracy and appropriate application of concepts in this domain
- Completeness (20%): Coverage of all required elements in the question
- Clarity (10%): Organization, coherence, and appropriate use of technical language

SCORING GUIDELINES:
- Full marks: Complete, correct answer demonstrating thorough understanding
- 70-90%: Mostly correct with minor omissions or errors
- 50-70%: Partially correct with significant concepts missing or misunderstood
- 20-50%: Major conceptual errors but shows some understanding
- 0-20%: Incorrect, irrelevant, or fundamentally misunderstands the question

QUESTION-SPECIFIC GUIDANCE:
- For definitions/short answers (Q1-5): Award full marks only for complete, accurate answers with correct terminology
- For mathematical questions: Check correctness of formulas, calculations, and final values
- For conceptual explanations: Look for key principles and proper explanation of underlying concepts
- For diagrams/graphs: Evaluate accuracy, labels, and whether it correctly illustrates the concept

IMPORTANT: If handwriting is unclear, make reasonable inferences based on context, but don't award marks for indecipherable content.

First, provide your detailed evaluation for each question with:
1. A summary of what the student's answer covered
2. Key concepts that were correctly or incorrectly addressed
3. Clear justification for the score given

After your evaluation, provide a summary table with the following format:

<SCORE_TABLE>
Question,MaxMarks,MarksObtained
1,{questions["1"][1]},your_score_for_q1
2,{questions["2"][1]},your_score_for_q2
...and so on for each question
</SCORE_TABLE>

Make sure to include the exact delimiters <SCORE_TABLE> and </SCORE_TABLE> and follow the CSV format strictly.
Ensure MarksObtained are numbers that don't exceed MaxMarks and are rounded to 1 decimal place when appropriate."""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": f"You are an expert {domain_context} instructor with experience in grading technical exams. Provide detailed, fair assessments based on domain knowledge and the given criteria."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1500
    }

    # Save API request for audit
    with open(eval_output_dir / "api_request_log.txt", "w") as f:
        f.write(f"\n--- Evaluation Request at {datetime.now().isoformat()} ---\n")
        # Don't log the full prompt which contains the answers, just log that we made the request
        f.write("Sent evaluation request to DeepSeek API\n")
    
    try:
        # Add rate limiting
        time.sleep(1)  # 1 second delay between calls
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        evaluation_result = result["choices"][0]["message"]["content"].strip()
        
        # Save result
        with open(eval_output_dir / "evaluation_result.txt", "w") as f:
            f.write(evaluation_result)
            
        return evaluation_result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during evaluation: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def parse_evaluation_results(evaluation_result, eval_output_dir):
    """Extracts question-wise marks from evaluation results using a delimiter-based approach."""
    # First save the raw evaluation result
    with open(eval_output_dir / "evaluation_result.txt", "w") as f:
        f.write(evaluation_result)
    
    # Extract the table between delimiters
    table_match = re.search(r'<SCORE_TABLE>\s*(.*?)\s*</SCORE_TABLE>', 
                           evaluation_result, re.DOTALL)
    
    if not table_match:
        logger.error("Could not find score table in the evaluation result")
        # Create empty dataframe with zeros as fallback
        summary = []
        for q_no, (question_text, max_marks) in questions.items():
            logger.warning(f"Could not extract score for question {q_no}, assuming zero")
            summary.append([q_no, question_text, max_marks, 0])
    else:
        # Parse the CSV data
        table_content = table_match.group(1).strip()
        lines = table_content.split('\n')
        
        # Skip header line
        summary = []
        for line in lines[1:]:
            if not line.strip():
                continue
                
            parts = line.split(',')
            if len(parts) >= 3:
                q_no = parts[0].strip()
                try:
                    max_marks = float(parts[1].strip())
                    score = float(parts[2].strip())
                    
                    # Validate score doesn't exceed max marks
                    if score > max_marks:
                        logger.warning(f"Score for question {q_no} ({score}) exceeds max marks ({max_marks}), capping at max")
                        score = max_marks
                        
                    # Add question text from our questions dictionary
                    question_text = questions.get(q_no, ["Unknown question", max_marks])[0]
                    summary.append([q_no, question_text, max_marks, score])
                except ValueError as e:
                    logger.warning(f"Error parsing score for question {q_no}: {e}")
                    # Use values from our questions dictionary as fallback
                    question_text, max_marks = questions.get(q_no, ["Unknown question", 0])
                    summary.append([q_no, question_text, max_marks, 0])
            else:
                logger.warning(f"Malformed line in score table: {line}")
    
    # Create dataframe and add total row
    df = pd.DataFrame(summary, columns=["Question No.", "Question", "Max Marks", "Marks Obtained"])
    
    # Calculate total
    total_marks = df["Max Marks"].sum()
    obtained_marks = df["Marks Obtained"].sum()
    
    # Add a row for total
    total_row = pd.DataFrame([["Total", "", total_marks, obtained_marks]], 
                             columns=["Question No.", "Question", "Max Marks", "Marks Obtained"])
    df = pd.concat([df, total_row], ignore_index=True)
    
    return df

def create_output_directory(pdf_path, app_config=None):
    """Create an output directory for this PDF run."""
    # Get base filename without extension
    base_name = os.path.basename(pdf_path).split('.')[0]
    
    # Get output directory from app config or use default
    output_base = "output"
    if app_config and "general" in app_config:
        output_base = app_config["general"].get("output_directory", "output")
    
    # Create base directory for this PDF
    base_output_dir = Path(output_base) / base_name
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Determine the next run number
    existing_runs = [d for d in os.listdir(base_output_dir) 
                   if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith("run_")]
    
    if existing_runs:
        # Extract run numbers and find the highest
        run_numbers = [int(run.split("_")[1]) for run in existing_runs]
        next_run = max(run_numbers) + 1
    else:
        next_run = 1
    
    # Create run directory
    run_dir = base_output_dir / f"run_{next_run}"
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

def create_evaluation_directory(run_dir, eval_run):
    """Create directory for an evaluation run"""
    eval_dir = run_dir / f"evaluate_{eval_run}"
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir

def validate_pdf_path(pdf_path):
    """Validate the PDF path exists and is a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"The file {pdf_path} is not a PDF file")
    
    return pdf_path

def load_app_config():
    """Load application configuration if available, otherwise use default."""
    config_path = Path(__file__).parent / "config" / "app_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded application configuration: {config}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load application config: {e}. Using default configuration.")
    
    # Create a sample config file if it doesn't exist
    if not config_path.exists():
        # Ensure config directory exists
        os.makedirs(Path(__file__).parent / "config", exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(APP_CONFIG, f, indent=2)
            logger.info(f"Created sample application configuration file at {config_path}")
        
        # Check for old config files and log a message if they exist
        old_ocr_config = Path(__file__).parent / "config" / "ocr_config.json"
        old_eval_config = Path(__file__).parent / "config" / "eval_config.json"
        
        if old_ocr_config.exists() or old_eval_config.exists():
            logger.info("Found old configuration files. These have been replaced by app_config.json but are kept for reference.")
    
    return APP_CONFIG.copy()

def main():
    try:
        logger.info("Starting automated answer script grading")
        
        # Ensure questions config file exists
        questions_config_path = Path(__file__).parent / "config" / "questions.json"
        if not questions_config_path.exists():
            # Ensure config directory exists
            os.makedirs(Path(__file__).parent / "config", exist_ok=True)
            logger.info("Creating default questions configuration file...")
            default_questions = {
                "1": ["What are the two most common supervised tasks?", 1],
                "2": ["What is the purpose of a validation set?", 1],
                "3": ["How many model parameters are there in a linear regression problem with a single feature variable?", 1],
                "4": ["What is the AUC value of a perfect classifier?", 1],
                "5": ["Out of precision and recall, which one is more important for a spam email detection system?", 1],
                "6": [
                    "What is train-test-split? What do you understand by overfitting and underfitting of training data and how do you prevent them?",
                    5],
                "7": [
                    "What are bias and variance of a machine learning model? How do you reduce them? What is the bias-variance trade-off?",
                    5],
                "8": [
                    "Explain the cost-functions associated with linear regression and logistic regression problems. What are the general algorithms that are available to minimize the cost-functions?",
                    5],
                "9": [
                    "What is the confusion matrix and why is it important? In a classification problem, true negative = 82, false positive =3, false negative=5, true positive=10, determine the following: precision, recall, false negative rate, false positive rate?",
                    5],
                "10": [
                    "What is ROC and AUC? Draw a roc curve for perfect classifier, practical classifier, and random classifier? What do you understand about the precision-recall trade-off?",
                    5],
                "11": [
                    "Draw and explain a typical ROC curve for the following cases; each figure represents the probability distribution of negative and positive prediction as function of decision threshold of a classifier.",
                    5],
            }
            with open(questions_config_path, 'w') as f:
                json.dump(default_questions, f, indent=4)
            logger.info(f"Created default questions configuration file at {questions_config_path}")
        
        # Get PDF path from user
        answer_script_pdf = input("Enter the path to the answer script PDF: ")
        try:
            answer_script_pdf = validate_pdf_path(answer_script_pdf)
        except (FileNotFoundError, ValueError) as e:
            logger.error(e)
            sys.exit(1)
            
        # Load application configuration
        app_config = load_app_config()
        logger.info(f"Application configuration loaded")
            
        # Create main output directory for this run
        output_dir = create_output_directory(answer_script_pdf, app_config)
        logger.info(f"Created output directory: {output_dir}")
        
        # Extract answers from PDF (done once per run)
        logger.info("Extracting student answers...")
        try:
            student_answers = extract_answers_from_pdf(answer_script_pdf, output_dir, app_config)
            logger.info(f"Extracted answers from {len(student_answers)} pages")
        except Exception as e:
            logger.error(f"Failed to extract answers: {e}")
            sys.exit(1)
        
        # Get evaluation configuration from app config
        eval_config = app_config.get("evaluation", EVAL_CONFIG.copy())
        num_evaluations = eval_config.get("num_evaluations", EVAL_CONFIG["num_evaluations"])
        
        # Run multiple evaluations
        all_summaries = []
        for eval_run in range(1, num_evaluations + 1):
            # Create directory for this evaluation run
            eval_output_dir = create_evaluation_directory(output_dir, eval_run)
            logger.info(f"Created evaluation directory: {eval_output_dir}")
            
            # Evaluate answers
            logger.info(f"Running evaluation {eval_run}/{num_evaluations}...")
            try:
                logger.info(f"Using domain context for evaluation: {app_config['general'].get('domain_context', 'machine learning exam')}")
                evaluation_result = evaluate_answers(questions, student_answers, eval_output_dir, app_config)
                logger.info(f"Evaluation {eval_run} completed successfully")
            except Exception as e:
                logger.error(f"Failed to complete evaluation {eval_run}: {e}")
                continue
                
            # Generate summary
            logger.info(f"Generating evaluation summary for run {eval_run}...")
            df_summary = parse_evaluation_results(evaluation_result, eval_output_dir)
            all_summaries.append(df_summary)
            
            # Save summary to CSV
            csv_path = eval_output_dir / "evaluation_summary.csv"
            df_summary.to_csv(csv_path, index=False)
            logger.info(f"Summary saved to {csv_path}")
            
            # Print summary for this evaluation
            print(f"\nEvaluation {eval_run} Summary:")
            print(df_summary)
        
        # Generate aggregate summary across all evaluations (optional)
        if all_summaries:
            logger.info("Generating aggregate summary across all evaluations...")
            
            # Simple average of all evaluations
            aggregate_df = pd.DataFrame(columns=["Question No.", "Question", "Max Marks", "Average Marks"])
            
            # Take the first summary as a template
            template_df = all_summaries[0][:-1]  # Exclude the total row
            
            # Calculate average for each question
            for _, row in template_df.iterrows():
                q_no = row["Question No."]
                question = row["Question"]
                max_marks = row["Max Marks"]
                
                # Get marks for this question across all evaluations
                question_marks = [df.loc[df["Question No."] == q_no, "Marks Obtained"].values[0] 
                                for df in all_summaries if q_no in df["Question No."].values]
                
                # Calculate average
                avg_marks = sum(question_marks) / len(question_marks) if question_marks else 0.0
                
                # Add to aggregate DataFrame
                aggregate_df = pd.concat([aggregate_df, pd.DataFrame([{
                    "Question No.": q_no,
                    "Question": question,
                    "Max Marks": max_marks,
                    "Average Marks": round(avg_marks, 2)
                }])], ignore_index=True)
            
            # Calculate total
            total_max = aggregate_df["Max Marks"].sum()
            total_avg = aggregate_df["Average Marks"].sum()
            
            # Add total row
            aggregate_df = pd.concat([aggregate_df, pd.DataFrame([{
                "Question No.": "Total",
                "Question": "",
                "Max Marks": total_max,
                "Average Marks": round(total_avg, 2)
            }])], ignore_index=True)
            
            # Save aggregate summary
            aggregate_csv_path = output_dir / "aggregate_evaluation_summary.csv"
            aggregate_df.to_csv(aggregate_csv_path, index=False)
            logger.info(f"Aggregate summary saved to {aggregate_csv_path}")
            
            # Print aggregate summary
            print("\nAggregate Evaluation Summary:")
            print(aggregate_df)
        
        print(f"\nAll results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
