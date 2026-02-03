from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data="../datasets/road_damage.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    name="yolov8l_road_damage"
)
