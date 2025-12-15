import csv
import os
import random

LAT_MIN, LAT_MAX = 48.84, 48.90
LON_MIN, LON_MAX = 2.29, 2.41

TYPES = ["nid_de_poule", "debris", "animal", "vehicule_arrete", "travaux"]
VOLUME_RANGE = (0.02, 1.80)

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    default_img = "images/demo.jpg"
    if not os.path.exists(default_img):
        default_img = ""

    rows = []
    for i in range(25):
        rows.append({
            "id": f"obs_{i:03d}",
            "latitude": round(random.uniform(LAT_MIN, LAT_MAX), 6),
            "longitude": round(random.uniform(LON_MIN, LON_MAX), 6),
            "volume": round(random.uniform(*VOLUME_RANGE), 3),
            "type_objet": random.choice(TYPES),
            "image_path": default_img
        })

    with open("data/obstacles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("CSV généré : data/obstacles.csv")

if __name__ == "__main__":
    main()
