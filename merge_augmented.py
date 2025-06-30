import os
import shutil

# Source and destination paths
source_dir = 'C:/Brain_tumor_Detection_Using_Federated_learning-main/clients/C1/Training/yes_augmented'
dest_dir = 'C:/Brain_tumor_Detection_Using_Federated_learning-main/clients/C1/Training/yes'

# Create dest dir if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Move each file from source to destination
for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))

print("✅ All augmented images moved to yes/ folder.")
