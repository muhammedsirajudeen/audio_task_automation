
from PIL import Image
import pytesseract
import spacy

# Load the pre-trained spaCy model
# If you're on Windows, specify the tesseract.exe path as below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open an image file
image_path = 'test.png'  # Replace with your image path
img = Image.open(image_path)

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(img)

# Print the extracted text
print(text)

nlp_custom = spacy.load("custom_name_model")

# Test the model
doc = nlp_custom(text)
names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
print(names)
