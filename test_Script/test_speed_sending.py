import requests
import os
import time
import csv

NGROK_URL = "https://pertly-nonhygrometric-sonny.ngrok-free.dev/upload"
IMAGE_FOLDER = "images"
CSV_FILE = "transmission_results.csv"

def run_batch_speed_test_to_csv():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg')]
    
    if not image_files:
        print("No .jpg images found.")
        return

    # Prepare the CSV file and write the header
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Size_KB", "Time_Seconds", "Speed_KB_s", "Status"])

        print(f"Testing {len(image_files)} images. Results will be saved to {CSV_FILE}")
        print("-" * 50)

        for filename in image_files:
            image_path = os.path.join(IMAGE_FOLDER, filename)
            file_size_kb = os.path.getsize(image_path) / 1024
            
            try:
                with open(image_path, 'rb') as img:
                    files = {'file': (filename, img, 'image/jpeg')}
                    headers = {'ngrok-skip-browser-warning': 'true'}
                    
                    start_clock = time.time()
                    response = requests.post(NGROK_URL, files=files, headers=headers, timeout=10)
                    end_clock = time.time()
                    
                    duration = end_clock - start_clock
                    
                    if response.status_code == 200:
                        speed = file_size_kb / duration
                        writer.writerow([filename, round(file_size_kb, 2), round(duration, 4), round(speed, 2), "Success"])
                        print(f"Done: {filename} ({round(duration, 2)}s)")
                    else:
                        writer.writerow([filename, round(file_size_kb, 2), 0, 0, f"HTTP_{response.status_code}"])
                        print(f"Failed: {filename} (HTTP {response.status_code})")
            
            except Exception as e:
                writer.writerow([filename, round(file_size_kb, 2), 0, 0, f"Error: {str(e)[:30]}"])
                print(f"Connection Error on {filename}")
                # Optional: sleep a bit if a connection error happens to let the network reset
                time.sleep(2)

    print("-" * 50)
    print(f"Test complete. Please check {CSV_FILE} for the full data.")

if __name__ == "__main__":
    run_batch_speed_test_to_csv()
