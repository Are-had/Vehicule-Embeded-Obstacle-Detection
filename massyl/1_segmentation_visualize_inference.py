import cv2
import numpy as np
import os
import time
from glob import glob
import onnxruntime as ort

# --- CONFIGURATION ---
# Point this to your exported ONNX file
MODEL_PATH = 'massyl/seg_models/stdc813m_maxmiou_4.onnx' 
MODEL_NAME = os.path.basename(MODEL_PATH)

VIDEO_SOURCE = 'massyl/data/lost_and_found_left_od_optimized/valid/images/*.jpg'
# VIDEO_SOURCE = 'my_drive_video.mp4'

# Must match the resolution used during ONNX export
INPUT_SIZE = (1024, 512) 

class InferenceDemo:
    def __init__(self):
        # 1. Load ONNX Model
        print(f"Loading ONNX model from {MODEL_PATH}...")
        
        # Check available providers (Prioritize CUDA if available, else CPU)
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        print(f"Inference Device: {providers[0]}")
        
        # Create Inference Session
        try:
            self.session = ort.InferenceSession(MODEL_PATH, providers=providers)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Get Input/Output Metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 3. Class info
        self.class_names = ["Background", "Road"]

        # BGR colors for OpenCV (Background=Blue/Black, Road=Green)
        self.colors = np.array([
            [255, 0, 0],   # Background
            [0, 255, 0],   # Road
        ], dtype=np.uint8)

    def preprocess(self, frame):
        """
        Preprocess image for ONNX Runtime (NumPy only, no Torch)
        """
        # 1. Resize
        frame_resized = cv2.resize(frame, INPUT_SIZE)
        
        # 2. BGR to RGB
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize (Manual implementation of torchvision.transforms.Normalize)
        # Convert to float32 and scale to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Mean and Std (ImageNet constants)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # (Image - Mean) / Std
        img_norm = (img_float - mean) / std
        
        # 4. Transpose HWC -> CHW (Channels, Height, Width)
        img_chw = img_norm.transpose(2, 0, 1)
        
        # 5. Add Batch Dimension [1, 3, H, W]
        input_tensor = np.expand_dims(img_chw, axis=0)
        
        return input_tensor, frame_resized

    def draw_legend(self, img, start_x=10, start_y=70):
        """
        Draw model name + class legend below FPS
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25

        # --- Model name ---
        cv2.putText(img, f"Model: {MODEL_NAME}", (start_x, start_y), 
                    font, font_scale, (255, 255, 255), thickness)

        y = start_y + line_height

        # --- Classes legend ---
        for idx, name in enumerate(self.class_names):
            color = self.colors[idx].tolist()
            # Color box
            cv2.rectangle(img, (start_x, y - 15), (start_x + 20, y + 5), color, -1)
            # Text
            cv2.putText(img, f"{idx}: {name}", (start_x + 30, y), 
                        font, font_scale, (255, 255, 255), thickness)
            y += line_height

    def run(self):
        # File Loading Logic
        if '*' in VIDEO_SOURCE:
            files = sorted(glob(VIDEO_SOURCE))
            print(f"Found {len(files)} images. Press 'q' to quit.")
            is_video = False
            if len(files) == 0:
                print("No images found! Check VIDEO_SOURCE path.")
                return
        else:
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            is_video = True
            print("Starting video. Press 'q' to quit.")

        idx = 0
        while True:
            start_time = time.time()

            if is_video:
                ret, frame = cap.read()
                if not ret: break
            else:
                if idx >= len(files): break
                frame = cv2.imread(files[idx])
                idx += 1

            if frame is None:
                continue

            # --- PREPROCESSING ---
            input_tensor, frame_resized = self.preprocess(frame)

            # --- INFERENCE (ONNX) ---
            # Run session (Input dictionary -> Output list)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # STDC/BiSeNet typically returns [output_map, aux_map_1, aux_map_2...]
            # We take the first one [0]
            raw_output = outputs[0] 
            
            # Post-processing: Argmax on Channel dimension (axis 1)
            # Input was [1, 2, H, W] -> Output [1, H, W]
            prediction = np.argmax(raw_output, axis=1).squeeze().astype(np.uint8)

            # --- DEBUG INFO ---
            unique_classes = np.unique(prediction)
            print(f"Frame {idx}: Classes found = {unique_classes}")
            
            total_pixels = prediction.size
            for c in unique_classes:
                count = np.count_nonzero(prediction == c)
                percent = (count / total_pixels) * 100
                if c < len(self.class_names):
                    print(f"   {self.class_names[c]} ({c}): {percent:.1f}%")
            print("-" * 20)

            # --- VISUALIZATION ---
            # Map class indices to colors
            seg_img = self.colors[prediction]
            
            # Overlay
            overlay = cv2.addWeighted(frame_resized, 0.7, seg_img, 0.3, 0)

            # FPS Calculation
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Legend
            self.draw_legend(overlay, start_x=10, start_y=70)

            cv2.imshow("Road Obstacle Detector (ONNX)", overlay)

            wait_time = 1 if is_video else 50
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break

        if is_video:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = InferenceDemo()
    demo.run()