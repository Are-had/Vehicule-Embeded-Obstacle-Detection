from ultralytics import YOLO
import os

# --- CONFIGURATION ---
# Path from your provided script
MODEL_PATH = "od_models/yolov8s_best_2.pt" 
IMG_SIZE = 640

def export_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading YOLO model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print("Starting ONNX export...")
    # Key arguments for Embedded/NPU deployment:
    # format='onnx'    : The target format
    # imgsz=IMG_SIZE   : Fixes the input size (Dynamic shapes are bad for AI HATs)
    # opset=11         : Version 11 is the most stable for Hailo/RPI AI HAT compilers
    # simplify=True    : Removes redundant operations to speed up inference
    path = model.export(
        format='onnx',
        imgsz=IMG_SIZE,
        opset=11,
        simplify=True
    )
    
    print(f"\n✅ Export success! Model saved to: {path}")
    print("You can now move this .onnx file to your Raspberry Pi.")

if __name__ == "__main__":
    export_model()