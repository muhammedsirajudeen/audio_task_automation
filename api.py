from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import  File, UploadFile,Form
import spacy

import os
# Initialize FastAPI instance
app = FastAPI()
from PIL import Image
import pytesseract

# Define a data model using Pydantic

app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join("static", "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/upload")
async def create_item(file: UploadFile = File(...)):
    file_location = f"uploaded_files/{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(await file.read())
    nlp_custom = spacy.load("custom_name_model")
    image_path = 'test.png'  # Replace with your image path
    img = Image.open(file_location)

# Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(img)
    doc = nlp_custom(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    print(names)
    return {"names":names}
    # return {"message": "success"}

@app.post('/uploadtext')
async def create_text(input: str = Form(...)):
    nlp_custom = spacy.load("custom_name_model")

    doc = nlp_custom(input)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    print(names)
    return {"names":names}

    # return {"message":"success","names":[]}
# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
