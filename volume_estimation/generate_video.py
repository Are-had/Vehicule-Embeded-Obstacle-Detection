import cv2
import glob
import os
from tqdm import tqdm

# Dossier depth maps
depth_folder = "data/test/depth/04_Maurener_Weg_8"

# Dossier output
output_folder = "data/test/video/04_Maurener_Weg_8"
os.makedirs(output_folder, exist_ok=True)

# Fichier vidéo
output_video = os.path.join(output_folder, "depth_video.mp4")

# FPS
fps = 5

# Lister images
depth_files = sorted(glob.glob(f"{depth_folder}/*_depth.png"))

print(f"Trouvé {len(depth_files)} images")

# Première image pour dimensions
first_frame = cv2.imread(depth_files[0])
height, width = first_frame.shape[:2]

# Créer vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Écrire frames
for depth_path in tqdm(depth_files):
    frame = cv2.imread(depth_path)
    out.write(frame)

out.release()

print(f"Vidéo sauvegardée: {output_video}")