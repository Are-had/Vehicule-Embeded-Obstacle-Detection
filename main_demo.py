import os
import pandas as pd
import shutil
import json
import time

# Configuration Paths
CSV_PATH = "demo_data/gps_data.csv"
LEFT_IMG_DIR = "demo_data/images_left"
RIGHT_IMG_DIR = "demo_data/images_right"
TEMP_BUFFER = "temp_buffer"

# Ensure the temporary buffer folder exists
if not os.path.exists(TEMP_BUFFER):
    os.makedirs(TEMP_BUFFER)

def prepare_demo_tasks():
    # 1. Load the GPS CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print("Error reading CSV file:", str(e))
        return

    print("Starting Demo Data Loader...")
    print(str(len(df)) + " frames found in CSV.")

    for index, row in df.iterrows():
        # Get data from CSV row
        # frame_name is '..._leftImg8bit.jpg'
        left_filename = str(row['frame_name'])
        lat = row['latitude']
        lon = row['longitude']

        # Determine the matching Right filename
        # This replaces the first 'left' with 'right'
        right_filename = left_filename.replace("left", "right")

        # Define full source paths
        left_src = os.path.join(LEFT_IMG_DIR, left_filename)
        right_src = os.path.join(RIGHT_IMG_DIR, right_filename)

        # Check if both files exist before moving to buffer
        if os.path.exists(left_src) and os.path.exists(right_src):
            # Create a unique subfolder for this specific capture
            # Using index to keep the sequence order
            task_id = "task_" + str(index).zfill(4)
            task_folder = os.path.join(TEMP_BUFFER, task_id)
            
            if not os.path.exists(task_folder):
                os.makedirs(task_folder)

            # Copy images to the buffer
            shutil.copy(left_src, os.path.join(task_folder, "left.jpg"))
            shutil.copy(right_src, os.path.join(task_folder, "right.jpg"))

            # Save GPS data to a JSON file inside the task folder
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

if __name__ == "__main__":
    prepare_demo_tasks()
