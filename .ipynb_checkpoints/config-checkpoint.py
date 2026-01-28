


# --- Configuration ---
ORIGINAL_DIR = "data_results/original"
PREDICTED_DIR = "data_results/predicted"
CSV_LOG = "data_results/detection_logs.csv"
WEIGHTS_DIR = "weights"


SPECIALIZED_MODELS = {
    "Obstacle_LAF":  {"path": "/home/imadb/Ghiles/runs/detect/train4/weights/best.pt", "color": (0, 128, 255)},
    "RoadWorks":     {"path": "/home/imadb/runs/detect/RoadWorks2/weights/best.pt", "color": (0, 255, 0)},
    "SpeedBumps":    {"path": "/home/imadb/runs/detect/SpeedBumps/weights/best.pt", "color": (255, 0, 0)},
    "StopSign":      {"path": "/home/imadb/runs/detect/StopSign/weights/best.pt", "color": (0, 0, 255)},
    "RoadDamage":    {"path": "/home/imadb/Ghiles/Projet_Ghiles/runs/detect/train4/weights/best.pt", "color": (255, 255, 0)},
    "RoadDebris1":   {"path": "/home/imadb/runs/detect/RoadDebris1/weights/best.pt", "color": (255, 0, 255)},
    "RoadObstacle":  {"path": "/home/imadb/runs/detect/road_obstacle_yolov8l/weights/best.pt", "color": (0, 200, 255)}
}


