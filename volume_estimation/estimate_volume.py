import cv2
import numpy as np
import json
import glob
import os
from tqdm import tqdm
import matplotlib.cm as cm

def process_disparities_to_depth(disparity_folder, camera_folder, output_folder):
    """
    Lit les disparités, calcule Z, et sauvegarde en PNG avec colormap jet
    
    Args:
        disparity_folder: Dossier contenant les *_disparity.png
        camera_folder: Dossier contenant les *_camera.json
        output_folder: Dossier pour sauvegarder les depth maps
    """
    
    # Créer dossier output si n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Lister tous les fichiers disparity
    disparity_files = sorted(glob.glob(f"{disparity_folder}/*_disparity.png"))
    
    print(f" Trouvé {len(disparity_files)} fichiers de disparité")
    
    # Colormap jet
    cmap = cm.get_cmap('jet')
    max_depth = 50  # Profondeur max en mètres
    
    # Traiter chaque fichier
    for disp_path in tqdm(disparity_files, desc="Calcul profondeur"):
        
        # 1. Trouver le fichier JSON correspondant
        base_name = os.path.basename(disp_path).replace('_disparity.png', '')
        json_path = os.path.join(camera_folder, f"{base_name}_camera.json")
        
        if not os.path.exists(json_path):
            print(f"  JSON manquant pour {base_name}")
            continue
        
        # 2. Charger disparité
        disp_raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        disparity = disp_raw.astype(np.float32) / 256.0  # Décoder
        
        # 3. Charger paramètres caméra
        with open(json_path, 'r') as f:
            camera_data = json.load(f)
        
        fx = camera_data['intrinsic']['fx']
        baseline = camera_data['extrinsic']['baseline']
        
        # 4. Calculer profondeur Z
        disparity_safe = disparity.copy()
        disparity_safe[disparity_safe <= 0] = np.nan
        
        depth = (fx * baseline) / disparity_safe
        
        # 5. Appliquer colormap jet
        depth_normalized = depth.copy()
        depth_normalized[depth_normalized > max_depth] = max_depth
        depth_normalized = depth_normalized / max_depth
        
        # Masque des pixels valides
        mask_valid = ~np.isnan(depth)
        
        # Appliquer jet colormap
        depth_colored = cmap(depth_normalized)[:, :, :3]  # RGB (0-1)
        depth_colored = (depth_colored * 255).astype(np.uint8)  # RGB (0-255)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)  # Pour OpenCV
        
        # Mettre pixels invalides en noir
        depth_colored[~mask_valid] = [0, 0, 0]
        
        # 6. Sauvegarder
        output_path = os.path.join(output_folder, f"{base_name}_depth.png")
        cv2.imwrite(output_path, depth_colored)
    
    print(f"\n Terminé ! {len(disparity_files)} depth maps sauvegardées dans {output_folder}")
    print(f" Format: PNG coloré avec colormap jet")


if __name__ == "__main__":
    
    # Chemins de tes dossiers
    disparity_folder = "test/disparity/04_Maurener_Weg_8"    # Dossier avec *_disparity.png
    camera_folder = "data/test/camera/04_Maurener_Weg_8"          # Dossier avec *_camera.json
    output_folder = "data/test/depth/04_Maurener_Weg_8"           # Où sauvegarder
    
    # Traiter tout
    process_disparities_to_depth(disparity_folder, camera_folder, output_folder)
    
    # Exemple de lecture d'un fichier généré
    print("TEST: Lecture d'un fichier depth")
    
    depth_files = glob.glob(f"{output_folder}/*_depth.png")
    if depth_files:
        test_file = depth_files[0]
        depth_img = cv2.imread(test_file)
        
        print(f"Fichier: {os.path.basename(test_file)}")
        print(f"Shape: {depth_img.shape}")