import requests
import os
import time
import csv
import cv2
import numpy as np

NGROK_URL = "https://pertly-nonhygrometric-sonny.ngrok-free.dev/upload"
IMAGE_FOLDER = "images"
CSV_FILE = "transmission_results_V2.csv"

def run_compressed_speed_test():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg')]
    
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Original_Size_KB", "Sent_Size_KB", "Time_Seconds", "Status"])

        for filename in image_files:
            path = os.path.join(IMAGE_FOLDER, filename)
            original_size = os.path.getsize(path) / 1024
            
            try:
                # 1. Load image using OpenCV
                img = cv2.imread(path)
                
                # 2. Optional: Resize (uncomment the line below to reduce dimensions by half)
                # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
                
                # 3. Compress to JPEG in memory (Quality 50 is much smaller but still clear)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                result, encimg = cv2.imencode('.jpg', img, encode_param)
                
                if result:
                    # Convert to byte stream for requests
                    img_byte_array = encimg.tobytes()
                    sent_size = len(img_byte_array) / 1024
                    
                    files = {'file': (filename, img_byte_array, 'image/jpeg')}
                    headers = {'ngrok-skip-browser-warning': 'true'}
                    
                    start_clock = time.time()
                    response = requests.post(NGROK_URL, files=files, headers=headers, timeout=10)
                    end_clock = time.time()
                    
                    duration = end_clock - start_clock
                    
                    writer.writerow([filename, round(original_size, 2), round(sent_size, 2), round(duration, 4), "Success"])
                    print(f"Sent: {filename} | Compressed: {round(sent_size, 2)}KB | Time: {round(duration, 2)}s")
            
            except Exception as e:
                print(f"Error on {filename}: {e}")
                time.sleep(1)

if __name__ == "__main__":
    run_compressed_speed_test()
