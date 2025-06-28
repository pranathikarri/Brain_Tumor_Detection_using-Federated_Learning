from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = load_model('trained_global_model.h5')

def Brain_Tumor(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    result = "⚠️ Tumor detected." if predictions[0][1] > 0.6 else "✅ No Tumor detected."
    print(f"{img_path} → {result}")

# Folder containing test images
test_folder = "test_data/"

# Loop through images in the folder and predict
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        Brain_Tumor(os.path.join(test_folder, filename))
