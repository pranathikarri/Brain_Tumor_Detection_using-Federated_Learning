import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'clients/C1/Training/',
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_generator = val_datagen.flow_from_directory(
    'clients/C1/Validation/',
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
    color_mode='grayscale'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),  # <--- added dropout
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
model.fit(
    train_generator,
    epochs=40,   # increased epochs
    validation_data=validation_generator,
    callbacks=[early_stop]  # added early stopping
)

# Evaluate model performance
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"✅ Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"✅ Validation Loss: {val_loss:.4f}")

# Save the model
model.save('trained_global_model.h5')

# (Optional) Save with pickle too
with open('trained_global_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Training complete and model saved.")
