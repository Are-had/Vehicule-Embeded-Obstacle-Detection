from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import os
import shutil
import csv
from processor import InferenceEngine

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config & Paths ---
ORIGINAL_DIR = "data_results/original"
PREDICTED_DIR = "data_results/predicted"
LOG_CSV = "data_results/inference_logs.csv"

# YOUR FIXED DICTIONARY
SPECIALIZED_MODELS = {
    "Obstacle_LAF":  {"path": "/home/imadb/Ghiles/runs/detect/train4/weights/best.pt", "color": (0, 128, 255)},
    "RoadWorks":     {"path": "/home/imadb/runs/detect/RoadWorks2/weights/best.pt", "color": (0, 255, 0)},
    "SpeedBumps":    {"path": "/home/imadb/runs/detect/SpeedBumps/weights/best.pt", "color": (255, 0, 0)},
    "StopSign":      {"path": "/home/imadb/runs/detect/StopSign/weights/best.pt", "color": (0, 0, 255)},
    "RoadDamage":    {"path": "/home/imadb/Ghiles/Projet_Ghiles/runs/detect/train4/weights/best.pt", "color": (255, 255, 0)},
    "RoadDebris1":   {"path": "/home/imadb/runs/detect/RoadDebris1/weights/best.pt", "color": (255, 0, 255)},
    "RoadObstacle":  {"path": "/home/imadb/runs/detect/road_obstacle_yolov8l/weights/best.pt", "color": (0, 200, 255)}
}

# Create directories
for path in [ORIGINAL_DIR, PREDICTED_DIR]:
    os.makedirs(path, exist_ok=True)

# Create CSV header if it doesn't exist
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "lat", "lon", "objects", "original_img", "predicted_img"])

# Initialize Engine
engine = InferenceEngine(SPECIALIZED_MODELS, PREDICTED_DIR)

@app.post("/upload")
async def handle_upload(
    image_left: UploadFile = File(...),
    image_right: UploadFile = File(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
    frame_id: str = Form(...)
):
    # 1. Save Original
    local_orig = os.path.join(ORIGINAL_DIR, f"{frame_id}_orig.jpg")
    with open(local_orig, "wb") as buffer:
        shutil.copyfileobj(image_left.file, buffer)

    # 2. Run Multi-Model Inference
    found_objects, inf_ms, local_pred = engine.run_inference(local_orig, frame_id)

    # 3. Log to CSV
    objects_str = ", ".join(found_objects) if found_objects else "None"
    with open(LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frame_id, latitude, longitude, objects_str, local_orig, local_pred])

    print(f"Processed {frame_id} | Time: {inf_ms:.1f}ms | Objects: {objects_str}")
    
    return {"status": "success", "objects": found_objects}

@app.get("/obstacles")
def get_obstacles(limit: int = 50):
    if not os.path.exists(LOG_CSV):
        return {"status": "success", "data": []}

    data = []
    with open(LOG_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # row = {"frame_id":..., "lat":..., "lon":..., "objects":..., "original_img":..., "predicted_img":...}
            try:
                data.append({
                    "id": row.get("frame_id", ""),
                    "latitude": float(row.get("lat", 0) or 0),
                    "longitude": float(row.get("lon", 0) or 0),
                    "objects": row.get("objects", ""),
                    "original_img": row.get("original_img", ""),
                    "predicted_img": row.get("predicted_img", ""),
                })
            except:
                # si une ligne est corrompue, on la saute
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
    uvicorn.run(app, host="0.0.0.0", port=8888)