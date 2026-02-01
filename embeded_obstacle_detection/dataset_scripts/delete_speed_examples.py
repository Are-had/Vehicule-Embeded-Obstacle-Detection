import os

# Paths to your dataset splits
splits = ["dataset_obstacles_massyl/train", "dataset_obstacles_massyl/valid"]

for split in splits:
    img_dir = os.path.join(split, "images")
    lbl_dir = os.path.join(split, "labels")

    for img_name in os.listdir(img_dir):
        # Only target files starting with '0000'
        if img_name.startswith("0000"):
            # Build full paths
            img_path = os.path.join(img_dir, img_name)
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)

            # Delete image
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted image: {img_path}")

            # Delete label if it exists
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
                print(f"Deleted label: {lbl_path}")
