import os
import shutil

# Source folder where augmented images are stored
source_folder = 'augmented_no_images'

# Destination folder (no class folder)
destination_folder = 'clients/C1/Training/no'

# Move each file from source to destination
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(source_path, destination_path)

print("All augmented images moved successfully.")
