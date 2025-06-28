from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv

# Load trained model
model = load_model('trained_global_model.h5')

def Brain_Tumor(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    result = "⚠️ Tumor detected." if predictions[0][1] > 0.6 else "✅ No Tumor detected."
    return result, predictions[0][0], predictions[0][1]

# Folder containing test images (flat folder — no subfolders inside)
test_folder = "test_data/"

# Output CSV file
output_file = "predictions.csv"

# Open CSV to write predictions
with open('predictions.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Result", "No Tumor Probability", "Tumor Probability"])

    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_folder, filename)
            result, no_prob, tumor_prob = Brain_Tumor(img_path)
            print(f"{filename} → {result}")
            writer.writerow([filename, result, round(no_prob, 4), round(tumor_prob, 4)])

print(f"\n✅ All predictions saved to {output_file}")
