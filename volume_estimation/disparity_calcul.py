import os
import cv2
import numpy as np
import json


# === PATHS ===
LEFT_DIR = "data/test/image_left"
RIGHT_DIR = "data/test/image_right"
CAMERA_FOLDER = "data/test/camera/04_Maurener_Weg_8"
DISPARITY_DIR = "outputs/disparity"
DEPTH_DIR = "outputs/depth"


os.makedirs(DISPARITY_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

# === FIND CAMERA JSON ===
def find_camera_json(image_filename, camera_folder):
    base_name = image_filename.replace("_leftImg8bit.jpg", "").replace("_leftImg8bit.png", "")
    camera_json = f"{base_name}_camera.json"
    json_path = os.path.join(camera_folder, camera_json)
    return json_path if os.path.exists(json_path) else None

# === LOAD CAMERA PARAMS ===
def load_camera_params(json_path):
    if not json_path or not os.path.exists(json_path):
        return None, None
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    fx = camera_data['intrinsic']['fx']
    baseline = camera_data['extrinsic']['baseline']
    return fx, baseline

# === COMPUTE DISPARITY ===
def compute_disparity(left_img, right_img):
    if left_img.shape != right_img.shape:
        right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
    
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    window_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*12,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

# === COMPUTE DEPTH ===
def compute_depth(disparity, fx, baseline):
    disparity[disparity <= 0] = 0.1
    depth = (fx * baseline) / disparity
    return depth

# === SAVE IMAGE (SEULEMENT JET COLORMAP) ===
def save_jet_colormap(data, output_path, is_depth=False):
    # Normalize
    if is_depth:
        data_clipped = np.clip(data, 0, 100)
        data_viz = cv2.normalize(data_clipped, None, 0, 255, cv2.NORM_MINMAX)
    else:
        data_viz = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    
    data_viz = np.uint8(data_viz)
    
    # Save JET colormap uniquement
    data_color = cv2.applyColorMap(data_viz, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, data_color)

# === PROCESS ===
left_images = sorted([f for f in os.listdir(LEFT_DIR) if f.endswith(('.jpg', '.png'))])

print(f"📂 Found {len(left_images)} images\n")

for left_filename in left_images:
    right_filename = left_filename.replace("leftImg8bit", "rightImg8bit")
    
    left_path = os.path.join(LEFT_DIR, left_filename)
    right_path = os.path.join(RIGHT_DIR, right_filename)
    
    if not os.path.exists(right_path):
        print(f"⚠️ Skip: {left_filename}\n")
        continue
    
    print(f"🔄 Processing: {left_filename}")
    
    # Find camera JSON
    camera_json_path = find_camera_json(left_filename, CAMERA_FOLDER)
    if camera_json_path:
        fx, baseline = load_camera_params(camera_json_path)
        print(f"   📷 fx={fx}, baseline={baseline}")
    else:
        fx, baseline = None, None
        print(f"   ⚠️ No camera JSON")
    
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"   ❌ Failed to load\n")
        continue
    
    # Compute disparity (SEULEMENT JET)
    disparity = compute_disparity(left_img, right_img)
    disparity_path = os.path.join(DISPARITY_DIR, f"disparity_{left_filename}")
    save_jet_colormap(disparity, disparity_path, is_depth=False)
    print(f"   ✅ Disparity saved")
    
    # Compute depth (SEULEMENT JET)
    if fx and baseline:
        depth = compute_depth(disparity, fx, baseline)
        depth_path = os.path.join(DEPTH_DIR, f"depth_{left_filename}")
        save_jet_colormap(depth, depth_path, is_depth=True)
        print(f"   ✅ Depth saved")
    
    print()

print("✅ All done!")