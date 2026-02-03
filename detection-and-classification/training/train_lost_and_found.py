from ultralytics import YOLO

# Utilisation de YOLOv8 Large pour une meilleure détection des petits objets
model = YOLO("yolov8l.pt")

model.train(
    data="../datasets/lost_and_found.yaml",
    epochs=150,
    imgsz=640,
    batch=16, # Si erreur "Out of Memory", baisse à 8
    device=0,
    name="yolov8l_lost_and_found"
)
