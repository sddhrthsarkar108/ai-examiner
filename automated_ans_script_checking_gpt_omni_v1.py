import openai
import base64
import os
import time
from pdf2image import convert_from_path

# OpenAI API Key should be set in the environment variable OPENAI_API_KEY

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
}


def encode_image_to_base64(image):
    """Convert an image file to base64 format."""
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text_from_image(image_path, max_retries=3):
    """Extract handwritten text from an image using OpenAI GPT-4o with retry mechanism."""
    base64_image = encode_image_to_base64(image_path)
    client = openai.OpenAI()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an OCR system that extracts handwritten answers from images."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract all handwritten answers from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OCR extraction failed (Attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)  # Wait before retrying

    print(f"Skipping image {image_path} after {max_retries} failed attempts. Saving for review.")
    os.rename(image_path, f"unreadable_{os.path.basename(image_path)}")
    return "[OCR Failed: Review manually]"


def extract_answers_from_pdf(pdf_path):
    """Convert a multi-page scanned PDF to images and extract handwritten text."""
    images = convert_from_path(pdf_path)
    extracted_text = ""

    for i, image in enumerate(images):
        image_path = f"temp_page_{i + 1}.jpg"
        image.save(image_path, "JPEG")
        text = extract_text_from_image(image_path)
        extracted_text += f"\nPage {i + 1}:\n{text}\n"
        os.remove(image_path)

    return extracted_text.strip()


def evaluate_answers(questions, answers):
    """Evaluate answers based on clarity, completeness, accuracy, examples, and presentation."""
    prompt = f"""Evaluate the following student answers based on clarity, completeness, accuracy, examples, and presentation.
    Give a score out of the total marks assigned to each question.

    Questions and Marks:
    {questions}

    Student Answers:
    {answers}

    Provide the score and a short justification for each question."""

    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an expert examiner who evaluates student answers based on predefined criteria."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return "Evaluation failed. Please retry."


def main():
    answer_script_pdf = input("Enter the path to the answer script PDF: ")
    print("Extracting student answers...")
    student_answers = extract_answers_from_pdf(answer_script_pdf)
    print("Extracted Answers:\n", student_answers)

    print("Evaluating answers...")
    evaluation_result = evaluate_answers(questions, student_answers)
    print("Evaluation Result:\n", evaluation_result)


if __name__ == "__main__":
    main()
