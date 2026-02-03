from ultralytics import YOLO

# On utilise le Large pour bien distinguer les cônes des autres objets
model = YOLO("yolov8l.pt")

model.train(
    data="../datasets/road_work.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    name="yolov8l_road_work"
)
