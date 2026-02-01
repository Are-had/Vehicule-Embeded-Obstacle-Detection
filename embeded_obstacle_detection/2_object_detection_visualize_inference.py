import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# --- CONFIGURATION ---
MODEL_PATH = "massyl/od_models/yolov8s_best_2.onnx"
IMAGE_DIR = "massyl/data/lost_and_found_left_od_optimized/valid/images"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

class YOLODetector:
    def __init__(self, model_path):
        print(f"Loading ONNX model from {model_path}...")
        
        # Initialize ONNX Runtime
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.img_size = self.input_shape[2] # Assuming square 640x640

        # Generate random colors for bounding boxes
        self.colors = np.random.uniform(0, 255, size=(100, 3))

    def preprocess(self, img):
        """
        Resize image to 640x640 with "Letterboxing" (padding) to maintain aspect ratio.
        This matches exactly what Ultralytics YOLO does internally.
        """
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (self.img_size, self.img_size)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        # Add border (padding)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # HWC to CHW, BGR to RGB, Normalize
        img_input = img_padded.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x640x640
        img_input = np.ascontiguousarray(img_input)
        img_input = img_input.astype(np.float32) / 255.0
        img_input = img_input[None]  # Add batch dimension: 1x3x640x640

        return img_input, (ratio, (dw, dh))

    def postprocess(self, output, dwdh, ratio, orig_shape):
        """
        Parse raw YOLO output: [1, 4+classes, 8400]
        Apply NMS and rescale boxes to original image.
        """
        # 1. Remove Batch Dimension
        # output[0] is a list containing the result tensor. 
        # The tensor shape is (1, 5, 8400). We need (5, 8400).
        prediction = output[0][0]
        
        # 2. Transpose to (8400, 5) so each row is a detection
        predictions = np.transpose(prediction, (1, 0))
        
        boxes = []
        scores = []
        class_ids = []

        # Separate coordinates and scores
        # Grid format: xc, yc, w, h, class_score...
        pred_boxes = predictions[:, :4]
        pred_scores = predictions[:, 4:]
        
        # Get max score and class ID for each anchor
        max_scores = np.max(pred_scores, axis=1)
        max_ids = np.argmax(pred_scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= CONF_THRESHOLD
        filtered_boxes = pred_boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_ids = max_ids[mask]

        if len(filtered_boxes) == 0:
            return [], [], []

        # Convert Center-x, Center-y, W, H -> Top-left X, Top-left Y, W, H
        # This is required for cv2.dnn.NMSBoxes
        nms_boxes = []
        for box in filtered_boxes:
            cx, cy, w, h = box
            x = cx - (w / 2)
            y = cy - (h / 2)
            nms_boxes.append([x, y, w, h])

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(nms_boxes, filtered_scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
        
        final_boxes = []
        final_scores = []
        final_ids = []

        if len(indices) > 0:
            for i in indices.flatten():
                # Get the box in 640x640 coords
                x, y, w, h = nms_boxes[i]
                
                # Rescale back to original image
                pad_w, pad_h = dwdh
                scale_w, scale_h = ratio
                
                x_orig = (x - pad_w) / scale_w
                y_orig = (y - pad_h) / scale_h
                w_orig = w / scale_w
                h_orig = h / scale_h
                
                # Convert to integer [x1, y1, x2, y2]
                x1 = int(x_orig)
                y1 = int(y_orig)
                x2 = int(x_orig + w_orig)
                y2 = int(y_orig + h_orig)
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_shape[1], x2)
                y2 = min(orig_shape[0], y2)

                final_boxes.append([x1, y1, x2, y2])
                final_scores.append(filtered_scores[i])
                final_ids.append(filtered_ids[i])

        return final_boxes, final_scores, final_ids

    def draw_detections(self, img, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = self.colors[class_id % len(self.colors)]
            x1, y1, x2, y2 = box
            
            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label = f"Obstacle {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    detector = YOLODetector(MODEL_PATH)

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if not image_files:
        print("No images found.")
        return

    print("q = previous | d = next | esc = quit")

    idx = 0
    n = len(image_files)

    while True:
        img_path = os.path.join(IMAGE_DIR, image_files[idx])
        frame = cv2.imread(img_path)

        if frame is None:
            idx = (idx + 1) % n
            continue

        # --- 1. PREPROCESS ---
        t0 = time.time()
        input_tensor, params = detector.preprocess(frame)

        # --- 2. INFERENCE ---
        # Run the model
        outputs = detector.session.run([detector.output_name], {detector.input_name: input_tensor})
        
        # --- 3. POSTPROCESS ---
        boxes, scores, class_ids = detector.postprocess(outputs, params[1], params[0], frame.shape)
        t1 = time.time()

        # --- 4. VISUALIZE ---
        detector.draw_detections(frame, boxes, scores, class_ids)

        inf_ms = (t1 - t0) * 1000

        # Nicer green + semi-transparent background + smaller size
        text = f"{image_files[idx]}  |  {inf_ms:.1f} ms"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4              # smaller than your current 0.8
        thickness = 1
        text_color = (80, 255, 120)    # pleasant bright green
        bg_color = (30, 30, 30)        # dark semi-transparent bg

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Position: top-left with small margin
        x, y = 12, 28

        # Optional: very subtle border
        # cv2.rectangle(frame, (x-7,y-text_h-9), (x+text_w+7,y+7), (60,60,60), 1)

        # Draw text on top
        cv2.putText(frame, text,
                    (x, y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

        cv2.imshow("YOLOv8 ONNX Detection", frame)

        # Navigation Logic
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):        # previous
            idx = (idx - 1) % n
        elif key == ord('d'):      # next
            idx = (idx + 1) % n
        elif key == 27:            # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()