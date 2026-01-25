import os
import time
import shutil
import requests
import json

# Configuration
TEMP_BUFFER = "temp_buffer"
NGROK_URL = "https://pertly-nonhygrometric-sonny.ngrok-free.dev/upload"
CHECK_INTERVAL = 2  # Seconds to wait between checks

def send_task(task_folder):
    """Handles the network request for a specific folder."""
    left_path = os.path.join(task_folder, "left.jpg")
    right_path = os.path.join(task_folder, "right.jpg")
    gps_path = os.path.join(task_folder, "gps.json")

    try:
        # Load GPS data to send as metadata
        with open(gps_path, 'r') as f:
            gps_data = json.load(f)

        # Prepare files for the multipart request
        files = {
            'image_left': ('left.jpg', open(left_path, 'rb'), 'image/jpeg'),
            'image_right': ('right.jpg', open(right_path, 'rb'), 'image/jpeg')
        }
        
        # We send GPS data as part of the form data
        data = {
            'latitude': gps_data['lat'],
            'longitude': gps_data['lon'],
            'frame_id': gps_data['frame']
        }

        headers = {'ngrok-skip-browser-warning': 'true'}

        print(f"Attempting to send {os.path.basename(task_folder)}...")
        response = requests.post(NGROK_URL, files=files, data=data, headers=headers, timeout=15)

        if response.status_code == 200:
            print(f"Successfully sent {os.path.basename(task_folder)}")
            return True
        else:
            print(f"Server error {response.status_code} for {os.path.basename(task_folder)}")
            return False

    except Exception as e:
        print(f"Network error: {e}")
        return False

def start_worker():
    print("Background Sender Worker started. Monitoring 'temp_buffer'...")
    
    while True:
        # Get all subfolders in the buffer
        tasks = [os.path.join(TEMP_BUFFER, d) for d in os.listdir(TEMP_BUFFER) 
                 if os.path.isdir(os.path.join(TEMP_BUFFER, d))]

        if not tasks:
            # No data to send, wait a bit
            time.sleep(CHECK_INTERVAL)
            continue

        for task_folder in tasks:
            success = send_task(task_folder)
            
            if success:
                # STEP 2 LOGIC: Delete only if the send was successful
                print(f"Cleaning up: Deleting {task_folder}")
                shutil.rmtree(task_folder)
            else:
                # If it failed, we leave it in the folder to try again in the next loop
                print(f"Postponing {os.path.basename(task_folder)} for retry...")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    start_worker()
