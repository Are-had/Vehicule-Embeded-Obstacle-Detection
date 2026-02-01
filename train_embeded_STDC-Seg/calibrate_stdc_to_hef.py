import hailo_sdk_client as hsdk
from hailo_sdk_client import ClientRunner
import os
import numpy as np
import cv2
from glob import glob

# --- CONFIGURATION ---
MODEL_NAME = "stdc_road_seg"
ONNX_PATH = "seg_models/stdc813m_maxmiou_4.onnx" # Your exported ONNX
CALIB_DIR = "calib_imgs"                         # The folder created above
HW_ARCH = "hailo8l"                              # Pi 5 AI Kit Architecture

# STDC Input Specs
INPUT_SHAPE = (1, 3, 512, 1024) 

def get_calibration_data():
    """
    Feeds chunks of 8 images to the compiler.
    NOTE: We return raw images. Normalization happens inside the model layers 
    via the 'normalization1' command below.
    """
    images = sorted(glob(os.path.join(CALIB_DIR, "*.png")))
    batch_data = []
    
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 1. Resize to match model input exactly
        img = cv2.resize(img, (INPUT_SHAPE[3], INPUT_SHAPE[2])) # Resize to (1024, 512)
        
        # 2. Convert BGR to RGB (OpenCV default is BGR, Model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        batch_data.append(img)
        
        if len(batch_data) >= 8:
            # Return batch as float32 (required by SDK) but values are 0-255
            yield [np.array(batch_data).astype(np.float32)]
            batch_data = []

def compile():
    print(f"Loading ONNX: {ONNX_PATH}")
    runner = ClientRunner(hw_arch=HW_ARCH)
    runner.translate_onnx_model(ONNX_PATH, model_name=MODEL_NAME)

    print("Running Optimization (Quantization)...")
    # This maps the FP32 values to INT8 using your real road images
    runner.optimize(get_calibration_data())

    print("Compiling to HEF...")
    # Define Normalization: Mean=[123.675...], Std=[58.395...] (ImageNet stats * 255)
    # This matches the (0.485, 0.456, 0.406) used in your training.
    alls_lines = [
        'model_optimization_flavor(optimization_level=2)',
        'performance_param(compiler_optimization_level=max)',
        'normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])'
    ]
    
    hef = runner.compile(alls_script="\n".join(alls_lines))
    
    with open(f"{MODEL_NAME}.hef", "wb") as f:
        f.write(hef)
    
    print(f"Success! {MODEL_NAME}.hef generated.")

if __name__ == "__main__":
    compile()