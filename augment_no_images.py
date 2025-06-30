import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Setup augmentation generator
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

# Folder paths
source_dir = r'C:\Brain_tumor_Detection_Using_Federated_learning-main\clients\C2\Training\yes'
dest_dir = r'C:\Brain_tumor_Detection_Using_Federated_learning-main\clients\C2\Training\yes_augmented'


os.makedirs(dest_dir, exist_ok=True)

# Augment each image 10 times
for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = load_img(os.path.join(source_dir, filename))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=dest_dir, save_prefix='aug', save_format='jpg'):
            i += 1
            if i > 10:
                break

print("✅ Image augmentation completed.")
