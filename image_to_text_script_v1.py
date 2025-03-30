import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
import cv2
import matplotlib.pyplot as plt

# Load the EasyOCR Reader (English)
reader = easyocr.Reader(['en'])

# Read the input image
image_path = "/Users/saswatisarkar/Downloads/IMG_4003.jpg"  # Change to your image path
image = cv2.imread(image_path)

# Convert image to RGB (Matplotlib displays in RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform OCR on the image
results = reader.readtext(image)

# Draw bounding boxes and display the recognized text
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Draw rectangle around text
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Put recognized text
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Display the image with detected text
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()

# Print extracted text
extracted_text = [text for (_, text, _) in results]
print("\nRecognized Text:\n", "\n".join(extracted_text))
