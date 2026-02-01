import torch
import cv2
import numpy as np
import os
import time
from glob import glob
from torchvision import transforms
from models.model_stages import BiSeNet

# --- CONFIGURATION ---
MODEL_PATH = 'pths/stdc813m_maxmiou_4.pth'
MODEL_NAME = os.path.basename(MODEL_PATH)

VIDEO_SOURCE = 'data/lost_and_found_formatted/images/train/*.png'
# VIDEO_SOURCE = 'my_drive_video.mp4'

INPUT_SIZE = (1024, 512)

class InferenceDemo:
    def __init__(self):
        # 1. Load Model
        print("Loading model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = BiSeNet(
            backbone='STDCNet813',
            n_classes=2,
            use_boundary_8=True
        )

        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace(".bn.bn.", ".bn.")
            new_state_dict[k] = v

        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        self.model.to(self.device)
        self.model.eval()

        # 2. Preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ])

        # 3. Class info
        self.class_names = ["Background", "Road"]

        # BGR colors for OpenCV
        self.colors = np.array([
            [255, 0, 0],       # Background -> Black
            [0, 255, 0],     # Road       -> Green
        ], dtype=np.uint8)


    def preprocess(self, frame):
        frame_resized = cv2.resize(frame, INPUT_SIZE)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
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
        cv2.putText(
            img,
            f"Model: {MODEL_NAME}",
            (start_x, start_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        y = start_y + line_height

        # --- Classes legend ---
        for idx, name in enumerate(self.class_names):
            color = self.colors[idx].tolist()

            # Color box
            cv2.rectangle(
                img,
                (start_x, y - 15),
                (start_x + 20, y + 5),
                color,
                -1
            )

            # Text
            cv2.putText(
                img,
                f"{idx}: {name}",
                (start_x + 30, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )

            y += line_height

    def run(self):
        if '*' in VIDEO_SOURCE:
            files = sorted(glob(VIDEO_SOURCE))
            print(f"Found {len(files)} images. Press 'q' to quit.")
            is_video = False
        else:
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            is_video = True
            print("Starting video. Press 'q' to quit.")

        idx = 0
        while True:
            start_time = time.time()

            if is_video:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if idx >= len(files):
                    break
                frame = cv2.imread(files[idx])
                idx += 1

            input_tensor, frame_resized = self.preprocess(frame)

            with torch.no_grad():
                output = self.model(input_tensor)[0]
                prediction = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

                # Debug info
                unique_classes = np.unique(prediction)
                print(f"Frame {idx}: Classes found = {unique_classes}")

                total_pixels = prediction.size
                for c in unique_classes:
                    count = np.count_nonzero(prediction == c)
                    percent = (count / total_pixels) * 100
                    print(f"   {self.class_names[c]} ({c}): {percent:.1f}%")
                print("-" * 20)

            # Visualization
            seg_img = self.colors[prediction]
            overlay = cv2.addWeighted(frame_resized, 0.7, seg_img, 0.3, 0)

            # FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                overlay,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            # Legend (below FPS)
            self.draw_legend(overlay, start_x=10, start_y=70)

            cv2.imshow("Road Obstacle Detector", overlay)

            wait_time = 1 if is_video else 50
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        if is_video:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = InferenceDemo()
    demo.run()
