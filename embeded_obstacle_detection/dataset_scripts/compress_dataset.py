import os
import shutil
import cv2
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "lost_and_found_right_od"
OUTPUT_DIR = "lost_and_found_right_od_optimized"

# Target Size: (Width, Height)
# Original is 2048x1024 (2:1 ratio). 
# We use 640x320 to maintain aspect ratio perfectly.
TARGET_SIZE = (640, 320) 
JPG_QUALITY = 80  # 0 to 100 (80 is a good balance for training)

def create_dir_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

def process_dataset():
    print(f"🚀 Starting Compression: {INPUT_DIR} -> {OUTPUT_DIR}")
    create_dir_structure()
    
    # We define the subfolders we expect to find
    splits = ["train", "valid"]
    
    total_saved = 0
    
    for split in splits:
        img_dir = os.path.join(INPUT_DIR, split, "images")
        lbl_dir = os.path.join(INPUT_DIR, split, "labels")
        
        # Output dirs
        out_img_dir = os.path.join(OUTPUT_DIR, split, "images")
        out_lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        
        # Find all images (jpg or png)
        images = glob(os.path.join(img_dir, "*"))
        
        print(f"📂 Processing {split}: {len(images)} images...")
        
        for img_path in tqdm(images):
            # 1. READ & RESIZE IMAGE
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Resize
            img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # 2. SAVE AS JPG
            filename = os.path.basename(img_path)
            # Force extension to .jpg
            new_filename = os.path.splitext(filename)[0] + ".jpg"
            save_path = os.path.join(out_img_dir, new_filename)
            
            cv2.imwrite(save_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
            
            # 3. COPY LABEL (Unchanged)
            # We need to find the corresponding txt file
            # Original might have been .png.txt or just .txt. 
            # Our previous script standardized on [filename].txt
            
            lbl_name = os.path.splitext(filename)[0] + ".txt"
            src_lbl_path = os.path.join(lbl_dir, lbl_name)
            dst_lbl_path = os.path.join(out_lbl_dir, lbl_name)
            
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, dst_lbl_path)
                
            total_saved += 1

    print("\n✅ Compression Complete!")
    print(f"   New dataset created at: {OUTPUT_DIR}")
    print(f"   Resolution: {TARGET_SIZE}")
    print(f"   Format: JPG (Quality {JPG_QUALITY})")

if __name__ == "__main__":
    process_dataset()