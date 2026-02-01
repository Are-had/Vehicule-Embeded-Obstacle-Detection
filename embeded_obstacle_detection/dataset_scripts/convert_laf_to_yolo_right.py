import os
import json
import shutil
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT_DIR = "lost_and_found"
IMG_ROOT = os.path.join(ROOT_DIR, "rightImg8bit")
LBL_ROOT = os.path.join(ROOT_DIR, "gtCoarse")
OUTPUT_DIR = "lost_and_found_right_od"

# Map Lost&Found splits to YOLO splits
SPLIT_MAP = {
    "train": "train",
    "test": "valid"
}

# Obstacle label whitelist
OBSTACLE_LABELS = {
    "01","02","03","04","05","06","07","08","09","10","11","12","13",
    "14","15","16","17","18","19","20","21",
    "22","23","24","25","26","27","28",
    "29","31","34"
}

# -------------------------------------------------

def create_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

def normalize_bbox(polygon, w, h):
    pts = np.array(polygon)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    bw = x_max - x_min
    bh = y_max - y_min
    cx = x_min + bw / 2
    cy = y_min + bh / 2

    return f"0 {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"

def process():
    create_dirs()
    stats = {"images": 0, "labels": 0, "backgrounds": 0}

    for lf_split, yolo_split in SPLIT_MAP.items():
        img_split_dir = os.path.join(IMG_ROOT, lf_split)
        if not os.path.exists(img_split_dir):
            continue

        # Walk recursively (city folders)
        image_paths = []
        for root, _, files in os.walk(img_split_dir):
            for f in files:
                if f.endswith(".png"):
                    image_paths.append(os.path.join(root, f))

        print(f"📂 {lf_split}: {len(image_paths)} images")

        for img_path in tqdm(image_paths):
            base = os.path.basename(img_path).replace(".png", "")
            out_img = os.path.join(OUTPUT_DIR, yolo_split, "images", base + ".png")
            out_lbl = os.path.join(OUTPUT_DIR, yolo_split, "labels", base + ".txt")

            # Copy image ALWAYS
            shutil.copy2(img_path, out_img)

            # Try to find label (may not exist!)
            rel = os.path.relpath(img_path, img_split_dir)
            label_rel = rel.replace("_rightImg8bit.png", "_gtCoarse_polygons.json")
            label_path = os.path.join(LBL_ROOT, lf_split, label_rel)

            yolo_lines = []

            if os.path.exists(label_path):
                with open(label_path) as f:
                    data = json.load(f)

                img_w = data["imgWidth"]
                img_h = data["imgHeight"]

                for obj in data.get("objects", []):
                    if obj["label"] in OBSTACLE_LABELS:
                        yolo_lines.append(
                            normalize_bbox(obj["polygon"], img_w, img_h)
                        )

            # Write label file ALWAYS (even empty)
            with open(out_lbl, "w") as f:
                f.write("\n".join(yolo_lines))

            if yolo_lines:
                stats["labels"] += len(yolo_lines)
            else:
                stats["backgrounds"] += 1

            stats["images"] += 1

    print("\n✅ DONE")
    print(f"Images copied: {stats['images']}")
    print(f"Obstacle boxes: {stats['labels']}")
    print(f"Background images (no labels): {stats['backgrounds']}")

# -------------------------------------------------

if __name__ == "__main__":
    process()
