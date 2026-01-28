import os
import pandas as pd
import shutil
import json
import time
from decision_engine import should_send_data

# Configuration Paths
CSV_PATH = "demo_data/gps_data.csv"
LEFT_IMG_DIR = "demo_data/images_left"
RIGHT_IMG_DIR = "demo_data/images_right"
MASK_DIR = "demo_data/flagged_masks"
TEMP_BUFFER = "temp_buffer"

if not os.path.exists(TEMP_BUFFER):
    os.makedirs(TEMP_BUFFER)

def prepare_demo_tasks():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print("Error reading CSV file:", str(e))
        return

    print("Starting Demo Data Loader...")
    print(str(len(df)) + " frames found in CSV.")

    for index, row in df.iterrows():
        left_filename = str(row['frame_name'])
        lat = row['latitude']
        lon = row['longitude']

        if not should_send_data(left_filename):
            continue

        right_filename = left_filename.replace("left", "right")
        mask_filename = "mask_" + left_filename

        left_src = os.path.join(LEFT_IMG_DIR, left_filename)
        right_src = os.path.join(RIGHT_IMG_DIR, right_filename)
        mask_src = os.path.join(MASK_DIR, mask_filename)

        if os.path.exists(left_src) and os.path.exists(right_src) and os.path.exists(mask_src):
            task_id = "task_" + str(index).zfill(4)
            task_folder = os.path.join(TEMP_BUFFER, task_id)

            if not os.path.exists(task_folder):
                os.makedirs(task_folder)

            shutil.copy(left_src, os.path.join(task_folder, "left.jpg"))
            shutil.copy(right_src, os.path.join(task_folder, "right.jpg"))
            shutil.copy(mask_src, os.path.join(task_folder, "mask.jpg"))

            gps_info = {
                "lat": lat,
                "lon": lon,
                "frame": left_filename
            }
            with open(os.path.join(task_folder, "gps.json"), "w") as f:
                json.dump(gps_info, f)

            print("Done: " + left_filename + " -> Buffered in " + task_id)
        else:
            print("Warning: Files missing for " + left_filename)
            if not os.path.exists(left_src):
                print("   Missing Left: " + left_src)
            if not os.path.exists(right_src):
                print("   Missing Right: " + right_src)
            if not os.path.exists(mask_src):
                print("   Missing Mask: " + mask_src)

if __name__ == "__main__":
    prepare_demo_tasks()
