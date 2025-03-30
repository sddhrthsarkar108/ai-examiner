import openai
import base64
import pandas as pd
import os
from datetime import datetime

# Set up OpenAI client with API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
client = openai.OpenAI(api_key=api_key)

# Path to the attendance CSV file
ATTENDANCE_FILE = "attendance.csv"

# Dictionary of student names and roll numbers
STUDENT_DICT = {
    "John Doe": "101",
    "Jane Smith": "102",
    "Michael Johnson": "103",
    "Emily Davis": "104",
    "David Wilson": "105"
}

def encode_image_to_base64(image_path):
    """Convert an image file to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_names_from_image(image_path):
    """Extract handwritten names from an image using OpenAI GPT-4 Vision."""
    base64_image = encode_image_to_base64(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an OCR system that extracts handwritten names from images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the handwritten names from this attendance sheet."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300
        )

        extracted_text = response.choices[0].message.content.strip()
        extracted_names = extracted_text.split("\n")  # Assuming names are line-separated
        print(extracted_names)
        return extracted_names

    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return []

def update_attendance(image_path):
    """Update the attendance CSV file with extracted names."""
    extracted_names = extract_names_from_image(image_path)

    if not extracted_names:
        print("No names detected. Attendance not updated.")
        return

    # Load or create the attendance DataFrame
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE, dtype=str)
    else:
        df = pd.DataFrame(columns=["Name", "Roll Number"])

    # Get the current date as a new column
    today_date = datetime.now().strftime("%Y-%m-%d")
    if today_date not in df.columns:
        df[today_date] = ""

    # Mark attendance
    for name in extracted_names:
        roll_number = STUDENT_DICT.get(name)
        if roll_number:
            if roll_number in df["Roll Number"].values:
                df.loc[df["Roll Number"] == roll_number, today_date] = "Present"
            else:
                df = pd.concat([df, pd.DataFrame([{"Name": name, "Roll Number": roll_number, today_date: "Present"}])], ignore_index=True)

    # Save the updated CSV
    df.to_csv(ATTENDANCE_FILE, index=False)
    print(df)
    print(f"✅ Attendance updated successfully in {ATTENDANCE_FILE}.")

if __name__ == "__main__":
    image_path = input("📷 Enter the path of the scanned attendance sheet: ")
    update_attendance(image_path)
