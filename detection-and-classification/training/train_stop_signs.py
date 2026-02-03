from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data="../datasets/stop_signs.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    name="yolov8l_stop_signs"
)
