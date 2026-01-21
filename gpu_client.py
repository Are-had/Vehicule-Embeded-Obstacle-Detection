import requests
import os
import config

def upload_to_gpu(image_path):
    """
    Sends the image at the specified path to the remote GPU.
    """
    if not os.path.exists(image_path):
        print(f"File not found at: {image_path}")
        return None

    try:
        # Open the file from the direct path provided
        with open(image_path, 'rb') as f:
            # We use os.path.basename to send just the filename (e.g., frame_L_...jpg) 
            # while the file content is read from the full path.
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(
                config.GPU_URL, 
                files=files, 
                timeout=config.REQUEST_TIMEOUT
            )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Server Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"⚠️Network Error during upload: {e}")
        return None
