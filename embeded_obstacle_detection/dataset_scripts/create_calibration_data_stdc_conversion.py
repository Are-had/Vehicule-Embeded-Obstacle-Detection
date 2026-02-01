import os
import shutil
import random
from glob import glob

# Config
SOURCE_ROOT = "lost_and_found_left_od_optimized/train/images"  # "cityscapes_massyl/leftImg8bit/train" # Your source
DEST_DIR = "yolo_calib_imgs"
TARGET_COUNT = 1024

# 1. Collect all valid image paths recursively
all_images = glob(os.path.join(SOURCE_ROOT, "**", "*.jpg"), recursive=True)

# 2. Randomly sample 1024
if len(all_images) > TARGET_COUNT:
    selected_images = random.sample(all_images, TARGET_COUNT)
else:
    selected_images = all_images

# 3. Copy to destination
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

print(f"Copying {len(selected_images)} images to {DEST_DIR}...")
for src in selected_images:
    # Use unique name to avoid overwriting if filenames repeat across folders
    base_name = os.path.basename(src)
    parent_name = os.path.basename(os.path.dirname(src))
    new_name = f"{parent_name}_{base_name}"
    shutil.copy(src, os.path.join(DEST_DIR, new_name))

print("Done! You are ready for compilation.")