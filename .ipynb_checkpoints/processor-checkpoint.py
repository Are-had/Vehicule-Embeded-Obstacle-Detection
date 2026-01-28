import os
import cv2
import time
from ultralytics import YOLO

class InferenceEngine:
    def __init__(self, weights_config, predicted_dir):
        """
        weights_config: Dictionary with structure: 
        "ModelName": {"path": "...", "color": (B, G, R)}
        """
        self.predicted_dir = predicted_dir
        self.models_info = weights_config
        self.loaded_models = {}

        print("--- Loading Specialized Models into GPU ---")
        for name, info in weights_config.items():
            path = info["path"]
            if os.path.exists(path):
                # Load the model and move to GPU if available
                self.loaded_models[name] = YOLO(path)
                print(f" Loaded: {name}")
            else:
                print(f" Error: Weights not found for {name} at {path}")
        
        print(f"Initialization complete. {len(self.loaded_models)} models ready.")

    def run_inference(self, image_path, frame_id):
        img = cv2.imread(image_path)
        if img is None:
            return [], 0, None
            
        annotated_img = img.copy()
        start_time = time.time()
        all_labels = []

        # Run each model on the same image
        for name, model in self.loaded_models.items():
            # Get the color assigned to this specific model
            model_color = self.models_info[name]["color"]
            
            # Inference
            results = model.predict(img, conf=0.4, verbose=False)
            
            # Manual Drawing for Custom Colors
            for box in results[0].boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[int(box.cls)]
                conf = float(box.conf)
                
                # Draw Rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), model_color, 3)
                
                # Draw Label background
                label_text = f"{name}: {label} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), model_color, -1)
                
                # Draw Text
                cv2.putText(annotated_img, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                all_labels.append(label)

        inference_time = (time.time() - start_time) * 1000
        
        # Save the result
        save_path = os.path.join(self.predicted_dir, f"{frame_id}_detected.jpg")
        cv2.imwrite(save_path, annotated_img)
        
        return all_labels, inference_time, save_path