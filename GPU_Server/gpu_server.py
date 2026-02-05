from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import os
import shutil
import csv
import json
from processor import InferenceEngine
from volume_estimate import compute_disparity, compute_depth, estimate_object_volume, FX, FY, BASELINE
import cv2

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORIGINAL_DIR = "data_results/original"
LEFT_DIR = os.path.join(ORIGINAL_DIR, "left")
RIGHT_DIR = os.path.join(ORIGINAL_DIR, "right")
MASK_DIR = os.path.join(ORIGINAL_DIR, "mask")
PREDICTED_DIR = "data_results/predicted"
LOG_CSV = "data_results/inference_logs.csv"

SPECIALIZED_MODELS = {
    "Obstacle_LAF":  {"path": "/home/imadb/Ghiles/runs/detect/train4/weights/best.pt", "color": (0, 128, 255), "conf": 25}, 
    "RoadWorks":     {"path": "/home/imadb/runs/detect/RoadWorks2/weights/best.pt", "color": (0, 255, 0), "conf": 60}, 
    "SpeedBumps":    {"path": "/home/imadb/runs/detect/SpeedBumps/weights/best.pt", "color": (255, 0, 0), "conf": 90}, 
    "RoadDamage":    {"path": "/home/imadb/Ghiles/Projet_Ghiles/runs/detect/train4/weights/best.pt", "color": (255, 255, 0), "conf": 85},  
    "RoadDebris1":   {"path": "/home/imadb/runs/detect/RoadDebris1/weights/best.pt", "color": (255, 0, 255), "conf": 40}, 
    "RoadObstacle":  {"path": "/home/imadb/runs/detect/road_obstacle_yolov8l/weights/best.pt", "color": (0, 200, 255), "conf": 40}
}

for path in [LEFT_DIR, RIGHT_DIR, MASK_DIR, PREDICTED_DIR]:
    os.makedirs(path, exist_ok=True)

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "lat", "lon", "objects", "original_img", "predicted_img", "volumes"])

engine = InferenceEngine(SPECIALIZED_MODELS, PREDICTED_DIR)

@app.post("/upload")
async def handle_upload(
    image_left: UploadFile = File(...),
    image_right: UploadFile = File(...),
    image_mask: UploadFile = File(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
    frame_id: str = Form(...)
):
    local_left = os.path.join(LEFT_DIR, f"{frame_id}.jpg")
    with open(local_left, "wb") as buffer:
        shutil.copyfileobj(image_left.file, buffer)
    
    local_right = os.path.join(RIGHT_DIR, f"{frame_id}.jpg")
    with open(local_right, "wb") as buffer:
        shutil.copyfileobj(image_right.file, buffer)
    
    local_mask = os.path.join(MASK_DIR, f"{frame_id}.jpg")
    with open(local_mask, "wb") as buffer:
        shutil.copyfileobj(image_mask.file, buffer)

    found_objects, inf_ms, local_pred, all_bboxes = engine.run_inference(local_left, frame_id)
    
    volumes_str = "None"
    if all_bboxes:
        print(f"Computing volumes for {len(all_bboxes)} objects...")
        
        left_img = cv2.imread(local_left)
        right_img = cv2.imread(local_right)
        
        disparity = compute_disparity(left_img, right_img)
        depth_map = compute_depth(disparity, FX, BASELINE)
        
        volumes_list = []
        for bbox_info in all_bboxes:
            x1, y1, x2, y2 = bbox_info["x1"], bbox_info["y1"], bbox_info["x2"], bbox_info["y2"]
            
            width_m, height_m, distance_m, surface_m2 = estimate_object_volume(
                (x1, y1, x2, y2), depth_map, FX, FY
            )
            
            volumes_list.append(f"{bbox_info['label']}:{surface_m2:.3f}m²@{distance_m:.2f}m")
        
        volumes_str = " | ".join(volumes_list)

    objects_str = ", ".join(found_objects) if found_objects else "None"
    with open(LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frame_id, latitude, longitude, objects_str, local_left, local_pred, volumes_str])

    print(f"Processed {frame_id} | Time: {inf_ms:.1f}ms | Objects: {objects_str}")
    print(f"Volumes: {volumes_str}")
    
    return {"status": "success", "objects": found_objects, "volumes": volumes_str}

@app.get("/obstacles")
def get_obstacles(limit: int = 50):
    if not os.path.exists(LOG_CSV):
        return {"status": "success", "data": []}

    data = []
    with open(LOG_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    "id": row.get("frame_id", ""),
                    "latitude": float(row.get("lat", 0) or 0),
                    "longitude": float(row.get("lon", 0) or 0),
                    "objects": row.get("objects", ""),
                    "original_img": row.get("original_img", ""),
                    "predicted_img": row.get("predicted_img", ""),
                    "volumes": row.get("volumes", "None")
                })
            except:
                continue

            if len(data) >= limit:
                break

    return {"status": "success", "data": data}

@app.get("/files/{file_path:path}")
def get_file(file_path: str):
    full_path = os.path.join(".", file_path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(full_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8880)