# import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
from constants import IMG_SIZE, TEST_IMAGE_NAME

# Cargar el modelo entrenado
model = load_model('train_model/trained_model.h5')

# Cargar el mapeo de clases
with open('train_model/class_indexes.json', 'r') as f:
    class_indices = json.load(f)

# Invertir el diccionario para obtener un mapeo de índice a nombre de clase
idx_to_class = {v: k for k, v in class_indices.items()}

# Función para cargar y preprocesar una imagen
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))  # Redimensionar la imagen
    img_array = image.img_to_array(img)  # Convertir la imagen a un arreglo numpy
    img_array = np.expand_dims(img_array, axis=0)  # Expandir la dimensión para que sea compatible con el modelo
    img_array = img_array / 255.0  # Normalizar
    return img_array

# Hacer la predicción
def predict_skin_disease(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)  # Realizar la predicción
    class_idx = np.argmax(prediction, axis=1)[0]  # Obtener el índice de la clase con la mayor probabilidad
    return class_idx, prediction

# Uso de la función
img_path = f'train_model/{TEST_IMAGE_NAME}'
class_idx, prediction = predict_skin_disease(img_path)

# Mostrar la imagen y la predicción
plt.imshow(image.load_img(img_path))
plt.title(f'Predicción: {idx_to_class[class_idx]}')
plt.show()

print(f"Predicción de enfermedad: {idx_to_class[class_idx]}")
print(f'Probabilidades por clase: {prediction}')
