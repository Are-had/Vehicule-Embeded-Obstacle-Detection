import os
import cv2
import time
from ultralytics import YOLO

class InferenceEngine:
    def __init__(self, weights_config, predicted_dir):
        """
        weights_config: Dictionary with structure: 
        "ModelName": {"path": "...", "color": (B, G, R), "conf": float}
        """
        self.predicted_dir = predicted_dir
        self.models_info = weights_config
        self.loaded_models = {}
        
        print("--- Loading Specialized Models into GPU ---")
        for name, info in weights_config.items():
            path = info["path"]
            if os.path.exists(path):
                self.loaded_models[name] = YOLO(path)
                print(f" Loaded: {name}")
            else:
                print(f" Error: Weights not found for {name} at {path}")
        
        print(f"Initialization complete. {len(self.loaded_models)} models ready.")
    
    def run_inference(self, image_path, frame_id):
        img = cv2.imread(image_path)
        if img is None:
            return [], 0, None, []
            
        annotated_img = img.copy()
        start_time = time.time()
        all_labels = []
        all_bboxes = []  # NEW
        
        for name, model in self.loaded_models.items():
            model_color = self.models_info[name]["color"]
            model_conf = self.models_info[name].get("conf", 0.25) / 100.0
            
            results = model.predict(img, conf=model_conf, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[int(box.cls)]
                conf = float(box.conf)
                
                # Save bbox info
                all_bboxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label,
                    "confidence": conf,
                    "model": name
                })
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), model_color, 3)
                
                label_text = f"{name}: {label} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), model_color, -1)
                
                cv2.putText(annotated_img, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                all_labels.append(label)
        
        inference_time = (time.time() - start_time) * 1000
        
        save_path = os.path.join(self.predicted_dir, f"{frame_id}_detected.jpg")
        cv2.imwrite(save_path, annotated_img)
        
        return all_labels, inference_time, save_path, all_bboxes  