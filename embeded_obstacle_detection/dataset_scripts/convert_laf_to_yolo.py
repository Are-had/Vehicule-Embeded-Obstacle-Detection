import os
import json
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT_DIR = "lost_and_found"
IMG_ROOT = os.path.join(ROOT_DIR, "rightImg8bit")
LBL_ROOT = os.path.join(ROOT_DIR, "gtCoarse")
OUTPUT_DIR = "lost_and_found_right_od"

# MAPPING CONFIG
# We map 'test' to 'valid' because L&F test set has labels we can use for validation.
SPLIT_MAP = {
    "train": "train",
    "test": "valid"
}

# --- THE WHITELIST (Derived from your PDF) ---
# These are the specific label STRINGS from the JSON that represent "Road Obstacles".
# EXCLUDED: Humans (39-42), Non-Hazards (30,32,33,35-38), Background (free, ego vehicle).
OBSTACLE_LABELS = {
    # Crates & Boxes (Black, Gray, Blue, Green)
    "01", "02", "03", "04", "05", "06", "07", "08", 
    "09", "10", "11", "12", "13",
    # Car Parts & Misc (Exhaust, Pallet, Pylon, Mirror, Tire)
    "14", "15", "16", "17", "18", "19", "20", "21",
    # "Emotional" Hazards (Ball, Bicycle, Dog, Dummy, Bobby Cars)
    "22", "23", "24", "25", "26", "27", "28",
    # Debris (Cardboard box, Plastic bag, Styrofoam)
    "29", "31", "34"
}

def create_dir_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Clean start
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

def normalize_bbox(polygon, img_w, img_h):
    """Converts polygon points to normalized YOLO bbox (xc, yc, w, h)."""
    pts = np.array(polygon)
    x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    
    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + (w / 2)
    y_center = y_min + (h / 2)
    
    return f"0 {x_center/img_w:.6f} {y_center/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"

def process_dataset():
    create_dir_structure()
    print(f"🚀 Starting Conversion: {ROOT_DIR} -> {OUTPUT_DIR}")
    
    stats = {"files_processed": 0, "obstacles_found": 0, "empty_backgrounds": 0}

    for lf_split, yolo_split in SPLIT_MAP.items():
        src_split_path = os.path.join(IMG_ROOT, lf_split)
        if not os.path.exists(src_split_path):
            print(f"⚠️  Skipping {lf_split} (folder not found)")
            continue

        # L&F has nested city folders, we walk through them
        image_files = []
        for root, _, files in os.walk(src_split_path):
            for f in files:
                if f.endswith(".png"):
                    image_files.append(os.path.join(root, f))

        print(f"📂 Processing {lf_split} ({len(image_files)} images)...")

        for img_path in tqdm(image_files):
            # 1. Find Matching Label JSON
            # Img:   .../leftImg8bit/train/city/city_123_leftImg8bit.png
            # Label: .../gtCoarse/train/city/city_123_gtCoarse_polygons.json
            
            # Construct expected label path
            rel_path = os.path.relpath(img_path, src_split_path) # city/city_123...
            label_rel = rel_path.replace("_leftImg8bit.png", "_gtCoarse_polygons.json")
            label_path = os.path.join(LBL_ROOT, lf_split, label_rel)
            
            if not os.path.exists(label_path):
                continue # Skip if no label (shouldn't happen often)

            # 2. Parse JSON
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            img_h, img_w = data['imgHeight'], data['imgWidth']
            yolo_lines = []
            
            for obj in data.get('objects', []):
                label = obj['label']
                if label in OBSTACLE_LABELS:
                    line = normalize_bbox(obj['polygon'], img_w, img_h)
                    yolo_lines.append(line)
                    stats["obstacles_found"] += 1

            # 3. Save to Output
            # Define new filenames (flattening the structure)
            filename_base = os.path.basename(img_path).replace(".png", "")
            dest_img_path = os.path.join(OUTPUT_DIR, yolo_split, "images", filename_base + ".jpg")
            dest_lbl_path = os.path.join(OUTPUT_DIR, yolo_split, "labels", filename_base + ".txt")
            
            # Write Label File
            with open(dest_lbl_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            
            # Copy Image (Convert to JPG for efficiency? Optional, copying PNG is safer/faster)
            # We rename to .jpg in the path above, so we should physically convert if we want
            # strictly jpg, but YOLO supports png. Let's just copy as is to save conversion time
            # and rename extension to match if you prefer. 
            # Actually, let's keep it simple: Copy file, keep extension.
            shutil.copy2(img_path, dest_img_path.replace(".jpg", ".png"))
            
            # Fix the label file name to match the png
            os.rename(dest_lbl_path, dest_lbl_path.replace(".txt", ".png").replace(".png", ".txt"))

            if len(yolo_lines) == 0:
                stats["empty_backgrounds"] += 1
            stats["files_processed"] += 1

    print("\n✅ DONE!")
    print(f"   - Images Processed: {stats['files_processed']}")
    print(f"   - Obstacles Extracted: {stats['obstacles_found']}")
    print(f"   - Background Images (Safe Roads): {stats['empty_backgrounds']}")
    print(f"   (These empty images are VITAL for reducing false positives!)")

if __name__ == "__main__":
    process_dataset()