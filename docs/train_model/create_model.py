from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
from constants import IMG_SIZE, BATCH_SIZE, EPOCHS, TRAINING_DATA_PATH_DATASET, TEST_DATA_PATH_DATASET

# Preprocesamiento de las imágenes usando ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,  # Normalizar las imágenes
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar el dataset de entrenamiento
train_generator = train_datagen.flow_from_directory(
    f'train_model/{TRAINING_DATA_PATH_DATASET}',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Cargar el dataset de validación
validation_generator = test_datagen.flow_from_directory(
    f'train_model/{TEST_DATA_PATH_DATASET}',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Construir el modelo CNN
model = models.Sequential([    
    # Primero
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Número de clases
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Guardar el modelo entrenado
model.save('train_model/trained_model.h5')

# Guardar el mapeo de clases
with open('train_model/class_indexes.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# Graficar el progreso del entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión del Modelo durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
