from PIL import Image
import os

folder = 'clients/C1/Training/no'

for filename in os.listdir(folder):
    try:
        img = Image.open(os.path.join(folder, filename))
        img.verify()  # verify image integrity
        print(f"{filename} - OK ✅")
    except Exception as e:
        print(f"{filename} - BROKEN ❌ ({e})")
