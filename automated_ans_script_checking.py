import openai
import base64
import os
from pdf2image import convert_from_path
import re
from PyPDF2 import PdfReader

def encode_image_to_base64(image):
    """Convert an image file to base64 format."""
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text_from_image(image_path):
    """Extract handwritten text from an image using OpenAI GPT-4 Vision."""
    base64_image = encode_image_to_base64(image_path)
    client = openai.OpenAI()  # Create OpenAI client instance
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an OCR system that extracts handwritten answers from images."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all handwritten answers from this image."},
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
    images = convert_from_path(pdf_path, poppler_path="/opt/homebrew/Cellar/poppler/25.03.0/bin/")
    extracted_text = ""
    for i, image in enumerate(images):
        image_path = f"temp_page_{i + 1}.jpg"
        image.save(image_path, "JPEG")
        extracted_text += extract_text_from_image(image_path) + "\n"
        os.remove(image_path)  # Clean up temp image
    return extracted_text.strip()

def extract_questions_and_marks(pdf_path):
    """Extract questions and their marks from a question paper PDF."""
    reader = PdfReader(pdf_path)
    question_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    questions = re.findall(r"(\d+\.\s.*?)(\(\d+\))", question_text)
    return {q.strip(): int(m.strip("()")) for q, m in questions}

def evaluate_answers(questions, answers):
    """Evaluate answers based on clarity, completeness, accuracy, examples, and presentation."""
    prompt = f"""Evaluate the following student answers based on clarity, completeness, accuracy, examples, and presentation.
    Give a score out of the total marks assigned to each question.

    Questions and Marks:
    {questions}

    Student Answers:
    {answers}

    Provide the score and a short justification for each question."""
    client = openai.OpenAI()  # Create OpenAI client instance
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert examiner who evaluates student answers based on predefined criteria."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return ""

def main():

    question_paper_pdf = input("Enter the path to the question paper PDF: ")
    answer_script_pdf = input("Enter the path to the answer script PDF: ")

    print("Extracting questions and marks...")
    questions = extract_questions_and_marks(question_paper_pdf)
    print(questions)

    print("Extracting student answers...")
    student_answers = extract_answers_from_pdf(answer_script_pdf)
    print(student_answers)

    print("Evaluating answers...")
    evaluation_result = evaluate_answers(questions, student_answers)

    print("Evaluation Result:\n", evaluation_result)

if __name__ == "__main__":
    main()
