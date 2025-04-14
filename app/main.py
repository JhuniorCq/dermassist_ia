from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import json

# Inicializar FastAPI
app = FastAPI()

# Permitir solicitudes desde cualquier origen (o reemplaza con frontend URL)
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Puedes restringir esto a tu frontend
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Cargar el modelo y clases
model = load_model("model/trained_model.h5")

with open("model/class_indexes.json", "r") as f:
  class_indices = json.load(f)
  classes = {v: k for k, v in class_indices.items()}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  contents = await file.read()
  image = Image.open(io.BytesIO(contents)).resize((224, 224))
  image = np.array(image) / 255.0
  image = np.expand_dims(image, axis=0)
  
  prediction = model.predict(image)
  class_index = int(np.argmax(prediction))
  confidence = float(np.max(prediction))
  class_name = classes[class_index]
  
  return {
    "class_index": class_index,
    "class_name": class_name,
    "confidence": confidence
  }
