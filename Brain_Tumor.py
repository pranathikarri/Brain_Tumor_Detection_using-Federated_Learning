from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('trained_global_model.h5')

def Brain_Tumor(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction

    predictions = model.predict(img_array)
    print(f"Tumor probability: {predictions[0][1]:.4f}")

    print("Raw predictions:", predictions)

    # If using softmax output (2 neurons: [no, yes])
    if predictions[0][1] > 0.6:
        print("⚠️ Tumor detected.")
    else:
        print("✅ No Tumor detected.")

# Example: test with a no image
Brain_Tumor("clients/C2/Training/no/3 no.jpg")

# Example: test with a yes image
Brain_Tumor("clients/C2/Training/yes/Y2.jpg")



