import openai
import base64
import os
from pdf2image import convert_from_path
import pandas as pd
import re
import requests
from fpdf import FPDF
from dotenv import load_dotenv
import time

# Add before the API call:
time.sleep(1)  # 1second delay between calls

# Load environment variables
load_dotenv()

# Set DeepSeek API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

print(f"API Key loaded: {'*' * len(DEEPSEEK_API_KEY)}")


# Hardcoded questions dictionary
questions = {
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


def encode_image_to_base64(image):
    """Convert an image file to base64 format."""
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text_from_image(image_path):
    """Extract handwritten text from an image using OpenAI GPT-4 Vision."""
    base64_image = encode_image_to_base64(image_path)
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an OCR system that extracts handwritten answers and detects figures from images."},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "Extract all handwritten answers and mention if any figures or diagrams are present. Clearly identify which question numbers are being answered on this page."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return ""


def extract_answers_from_pdf(pdf_path):
    """Convert a multi-page scanned PDF to images and extract handwritten text."""
    images = convert_from_path(pdf_path, poppler_path="/opt/homebrew/bin/")
    extracted_answers = {}

    for i, image in enumerate(images):
        image_path = f"temp_page_{i + 1}.jpg"
        image.save(image_path, "JPEG")
        extracted_text = extract_text_from_image(image_path)
        extracted_answers[f"Page {i + 1}"] = extracted_text
        os.remove(image_path)  # Clean up temp image

    return extracted_answers


def detect_attempted_questions(extracted_answers):
    """
    Determine which questions have been attempted by the student.
    Returns a dictionary of question numbers with True/False values.
    """
    # Initialize all questions as unattempted
    attempted_questions = {str(q_no): False for q_no in questions.keys()}
    
    # Regular expressions to detect question numbers in various formats
    patterns = [
        r'Q(?:uestion)?\s*#?\s*(\d+)',      # Q1, Question 1, etc.
        r'(?:^|\n)\s*(\d+)\s*[.:]',        # 1., 1:, etc. at start of line
        r'Part\s+(?:A|B|C|D)[\s:]*(?:[Qq]uestion)?\s*(\d+)', # Part A: 1, Part B Question 3, etc.
        r'(?:^|\n)\s*(?:Part\s+(?:A|B))[\s:]*.*?(?:question|Q)?\s*(\d+)', # Part A: 1, etc.
        r'(?:^|\n)\s*(?:Part\s+(?:A|B))\s*.*?(?:\n|$).*?(?:(?:^|\n)\s*(\d+)[.:])', # Questions under parts
        r'[Aa]nswer\s*(?:to)?\s*(?:[Qq]uestion)?\s*#?\s*(\d+)', # Answer to Question 1
    ]
    
    # Check all extracted text
    all_text = ' '.join(extracted_answers.values())
    
    # Look for question keywords with numbers
    for pattern in patterns:
        matches = re.finditer(pattern, all_text)
        for match in matches:
            q_no = match.group(1)
            if q_no in attempted_questions:
                attempted_questions[q_no] = True
    
    # Additional specific checks for each question
    for q_no, (question_text, _) in questions.items():
        # Create key terms from the question to look for in answers
        key_terms = []
        if q_no == "1":
            key_terms = ["classification", "regression", "supervised"]
        elif q_no == "2":
            key_terms = ["validation set", "cross validation", "hyperparameter"]
        elif q_no == "3":
            key_terms = ["parameter", "linear regression", "feature"]
        elif q_no == "4":
            key_terms = ["AUC", "perfect classifier", "1.0", "ROC"]
        elif q_no == "5":
            key_terms = ["precision", "recall", "spam", "email"]
        elif q_no == "6":
            key_terms = ["train-test", "split", "overfitting", "underfitting"]
        elif q_no == "7":
            key_terms = ["bias", "variance", "trade-off", "trade off"]
        elif q_no == "8":
            key_terms = ["cost function", "linear regression", "logistic regression", "minimize"]
        elif q_no == "9":
            key_terms = ["confusion matrix", "precision", "recall", "false negative", "false positive"]
        elif q_no == "10":
            key_terms = ["ROC", "AUC", "curve", "precision-recall"]
        elif q_no == "11":
            key_terms = ["ROC curve", "probability distribution", "decision threshold"]
        
        # Check if key terms are present in the answers
        for term in key_terms:
            if term.lower() in all_text.lower():
                # If a question's key terms are found but not explicitly labeled, 
                # only mark as attempted if there's substantial content
                if not attempted_questions[q_no]:
                    # Count how many key terms are present
                    term_count = sum(1 for t in key_terms if t.lower() in all_text.lower())
                    if term_count >= len(key_terms) // 2:  # If at least half the terms are present
                        attempted_questions[q_no] = True
                        break
    
    # One more check: if answers have well-defined question numbering
    for page_text in extracted_answers.values():
        if "**Part A:**" in page_text or "**Part B:**" in page_text:
            part_sections = re.split(r'\*\*Part [A-Z]:\*\*', page_text)
            for section in part_sections:
                if not section.strip():
                    continue
                # Look for numbered items that are likely answers
                numbered_items = re.findall(r'(?:^|\n)\s*(\d+)\.\s+(.*?)(?=(?:\n\s*\d+\.)|$)', section, re.DOTALL)
                for num, content in numbered_items:
                    if num in attempted_questions and content.strip():
                        attempted_questions[num] = True
    
    print("\nAttempted Questions Analysis:")
    for q_no, attempted in attempted_questions.items():
        status = "Attempted" if attempted else "Not Attempted"
        print(f"Question {q_no}: {status}")
    
    return attempted_questions


def evaluate_answers(questions, answers, attempted_questions):
    """Evaluate answers based on clarity, completeness, accuracy, examples, and presentation."""
    prompt = f"""Evaluate the following student answers based on clarity, completeness, and accuracy.
    Give a score out of the total marks assigned to each question.
    
    Questions and Marks:
    {questions}

    Student Answers:
    {answers}
    
    Attempted Questions Analysis:
    {attempted_questions}
    
    IMPORTANT INSTRUCTIONS:
    1. Assign ZERO marks to any question that is marked as "Not Attempted" in the analysis above.
    2. Only evaluate questions that are actually attempted by the student.
    3. If a question is marked as "Attempted" but you cannot find clear evidence of an answer, assign ZERO marks.
    4. Be strict about assigning marks - don't give credit for content from other questions.

    Please provide your evaluation in the following format:
    Question|Score|Max Marks|Comments
    1|4|5|Good explanation but missing one point
    2|1|1|Correct answer
    ...

    For each question:
    1. Provide a score out of the maximum marks
    2. Include brief comments justifying the score
    3. Use the pipe (|) character as a delimiter
    4. Start the table with the header row exactly as shown above
    5. Include all questions in the table
    6. For unattempted questions, always assign a score of 0 and comment "Not attempted by student"
    """

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
                "content": "You are an expert examiner who evaluates student answers based on predefined criteria. Always format your response as a table with pipe-delimited columns. Be very strict about only giving marks to questions that have been attempted."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1500
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return ""


def parse_evaluation_results(evaluation_result):
    """Extracts question-wise marks from evaluation results and stores them in a DataFrame using delimiter-based parsing."""
    try:
        # Find the table in the response
        lines = evaluation_result.split('\n')
        table_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('Question|Score|Max Marks|Comments'):
                table_start = i
                break
        
        if table_start == -1:
            print("Warning: Could not find the evaluation table in the response")
            return pd.DataFrame(columns=["Question No.", "Question", "Max Marks", "Marks Obtained", "Comments"])
        
        # Extract table rows
        table_rows = []
        for line in lines[table_start + 1:]:
            if not line.strip() or not '|' in line:
                continue
            parts = line.split('|')
            if len(parts) >= 4:
                table_rows.append(parts)
        
        # Create DataFrame
        df = pd.DataFrame(table_rows, columns=["Question No.", "Marks Obtained", "Max Marks", "Comments"])
        
        # Convert numeric columns
        df["Marks Obtained"] = pd.to_numeric(df["Marks Obtained"], errors='coerce')
        df["Max Marks"] = pd.to_numeric(df["Max Marks"], errors='coerce')
        
        # Add question text from the questions dictionary
        df["Question"] = df["Question No."].map(lambda x: questions.get(str(x), [""])[0])
        
        # Reorder columns
        df = df[["Question No.", "Question", "Max Marks", "Marks Obtained", "Comments"]]
        
        return df
        
    except Exception as e:
        print(f"Error parsing evaluation results: {e}")
        return pd.DataFrame(columns=["Question No.", "Question", "Max Marks", "Marks Obtained", "Comments"])


def main():
    answer_script_pdf = input("Enter the path to the answer script PDF: ")
    output_pdf = "extracted_answers.pdf"

    print("Extracting student answers...")
    student_answers = extract_answers_from_pdf(answer_script_pdf)

    print(student_answers)
    
    print("Analyzing attempted questions...")
    attempted_questions = detect_attempted_questions(student_answers)

    print("Evaluating answers...")
    evaluation_result = evaluate_answers(questions, student_answers, attempted_questions)

    print("Evaluation Result:\n", evaluation_result)

    print("Generating summary...")
    df_summary = parse_evaluation_results(evaluation_result)
    print(df_summary)

    # Save to CSV or Excel if needed
    df_summary.to_csv("evaluation_summary.csv", index=False)
    print("Summary saved as evaluation_summary.csv")


if __name__ == "__main__":
    main()
