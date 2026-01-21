import time
import os
import config
import utils
from gps_handler import GPSModule

# Note: You will import your camera library here (e.g., import cv2)

def main():
    print(" PFE Capture System - Production Mode")
    
    # Initialize GPS
    gps = GPSModule()
    
    # Ensure data directory exists
    if not os.path.exists(config.DATA_LOG_PATH):
        os.makedirs(config.DATA_LOG_PATH)

    print(" System standby. Waiting for GPS fix...")

    try:
        while True:
            # 1. Update Sensors
            gps_status = gps.update()

            if gps_status["valid"]:
                # 2. Get Metadata
                lat, lon = gps_status["lat"], gps_status["lon"]
                sats = gps_status["used"]
                timestamp = time.strftime("%H%M%S")

                # 3. TRIGGER LOGIC 
                # (Later: if local_model.detect(frame): ...)
                should_capture = True 

                if should_capture:
                    # A. Generate filenames
                    name_L = utils.generate_filename("L", timestamp, lat, lon)
                    name_R = utils.generate_filename("R", timestamp, lat, lon)
                    path_L = os.path.join(config.DATA_LOG_PATH, name_L)
                    path_R = os.path.join(config.DATA_LOG_PATH, name_R)

                    # B. REAL CAMERA CAPTURE (Placeholders for your future hardware)
                    # Example for OpenCV:
                    # cv2.imwrite(path_L, frame_left)
                    # cv2.imwrite(path_R, frame_right)
                    print(f"Saved: {name_L} and {name_R}")

                    # C. Save Metadata Sidecar
                    utils.save_metadata(path_L, lat, lon, timestamp, sats)
                    utils.save_metadata(path_R, lat, lon, timestamp, sats)

            else:
                print(f"Searching Satellites... (Visible: {gps_status['detected']})", end="\r")
            
            time.sleep(0.1) # Loop frequency (10Hz)

    except KeyboardInterrupt:
        print("\n System Shut Down.")

if __name__ == "__main__":
    main()
