import os
import pandas as pd
import cv2
import shutil
import json

# Paths
CSV_PATH = "demo_data/gps_data.csv"
LEFT_IMG_DIR = "demo_data/images_left"
RIGHT_IMG_DIR = "demo_data/images_right"
TEMP_BUFFER = "temp_buffer"

# Ensure temp folder exists
os.makedirs(TEMP_BUFFER, exist_ok=True)

def should_send_data(lat, lon, frame_id):
    """
    Step 1 Logic: Decide if we should send this frame.
    Example: You could send every 5th frame, or only if moving.
    """
    # For now, let's say we send every frame for the demo
    return True

def prepare_data_for_sending():
    # 1. Load the GPS CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Starting Demo Data Loader... {len(df)} frames found.")

    for index, row in df.iterrows():
        # Assuming your CSV has columns: frame_name, latitude, longitude
        frame_name = row['frame_name'] 
        lat = row['latitude']
        lon = row['longitude']

        if should_send_data(lat, lon, frame_name):
            # Paths to the actual images
            left_path = os.path.join(LEFT_IMG_DIR, f"{frame_name}_left.jpg")
            right_path = os.path.join(RIGHT_IMG_DIR, f"{frame_name}_right.jpg")

            if os.path.exists(left_path) and os.path.exists(right_path):
                # Create a unique subfolder in temp_buffer for this specific capture
                capture_id = f"cap_{index}_{int(os.time.time())}"
                task_folder = os.path.join(TEMP_BUFFER, capture_id)
                os.makedirs(task_folder, exist_ok=True)

                # 1. Copy Images to Temp
                shutil.copy(left_path, os.path.join(task_folder, "left.jpg"))
                shutil.copy(right_path, os.path.join(task_folder, "right.jpg"))

                # 2. Save GPS as a small JSON file
                gps_info = {"lat": lat, "lon": lon, "frame": frame_name}
                with open(os.path.join(task_folder, "gps.json"), "w") as f:
                    json.dump(gps_info, f)

                print(f"Prepared Task: {capture_id} (Lat: {lat}, Lon: {lon})")
            else:
                print(f"Warning: Images for {frame_name} not found.")

if __name__ == "__main__":
    prepare_data_for_sending()
