from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = load_model('models/best_model.h5')

# Class names and descriptions
class_map = {
    0: {"name": "Melanoma", "description": "A type of skin cancer that begins in melanocytes, the pigment-producing cells."},
    1: {"name": "Seborrheic Keratosis", "description": "A non-cancerous (benign) tumor that originates from cells that make the outer layer of skin."},
    2: {"name": "Basal Cell Carcinoma", "description": "A common type of skin cancer that originates in basal cells, typically found in sun-exposed areas."},
    3: {"name": "Squamous Cell Carcinoma", "description": "A type of skin cancer that begins in squamous cells, often in sun-exposed areas of the skin."},
    4: {"name": "Vascular Lesion", "description": "Abnormal growth of blood vessels, often benign, can appear as red or purple spots on the skin."},
    5: {"name": "Nevus", "description": "Commonly known as a mole, it’s a growth on the skin that’s typically benign but can become malignant."},
    6: {"name": "Dermatofibroma", "description": "A benign skin tumor that usually appears as a firm, brownish bump on the skin."}
}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    pred_index = np.argmax(model.predict(image))
    
    # Get the class name and description from the class_map
    prediction = class_map[pred_index]
    
    return {"prediction": prediction}
