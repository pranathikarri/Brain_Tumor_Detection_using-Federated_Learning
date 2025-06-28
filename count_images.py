import os

# Set the path to your training directory
train_dir = 'clients/C1/Training'

# List the class folders (e.g. ['no', 'yes'])
class_folders = os.listdir(train_dir)

for class_name in class_folders:
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Class '{class_name}' has {num_images} images.")
