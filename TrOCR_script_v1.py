from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# Load model & processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Load image
#image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Handwriting_Sample.jpg/800px-Handwriting_Sample.jpg"  # Replace with your image path

image_path = "/Users/saswatisarkar/Downloads/ca3_answer.pdf"  # Use local file path
image = Image.open(image_path).convert("RGB")


# Process & predict text
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Recognized Text:", recognized_text)
