import json, random, shutil
from pathlib import Path
import cv2

# 1 seule classe YOLO : obstacle
CLASS_ID = 0

# NON obstacles (d’après le tableau Lost&Found) :contentReference[oaicite:1]{index=1}
NON_OBS = {"unlabeled","ego vehicle","rectification border","out of roi","background","free",
           "30","32","33","35","36","37","38"}  # trainID=0

def poly2bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def bbox2yolo(x1,y1,x2,y2,W,H):
    bw = max(0, x2-x1); bh = max(0, y2-y1)
    cx = x1 + bw/2; cy = y1 + bh/2
    return cx/W, cy/H, bw/W, bh/H

def json_for(img_path, gt_root, split):
    seq = img_path.parent.name
    stem = img_path.name.replace("_leftImg8bit.png","")
    return gt_root/split/seq/f"{stem}_gtCoarse_polygons.json"

def convert_split(img_root, gt_root, out_images, out_labels, split):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_path in img_root.rglob("*_leftImg8bit.png"):
        jpath = json_for(img_path, gt_root, split)
        if not jpath.exists(): 
            continue

        img = cv2.imread(str(img_path))
        if img is None: 
            continue
        H,W = img.shape[:2]

        data = json.loads(Path(jpath).read_text(encoding="utf-8"))
        objs = data.get("objects", [])

        lines = []
        for o in objs:
            label = str(o.get("label",""))
            poly = o.get("polygon", None)
            if label in NON_OBS or poly is None or len(poly) < 3:
                continue
            x1,y1,x2,y2 = poly2bbox(poly)
            x1 = max(0,min(int(x1),W-1)); x2 = max(0,min(int(x2),W-1))
            y1 = max(0,min(int(y1),H-1)); y2 = max(0,min(int(y2),H-1))
            if (x2-x1)*(y2-y1) < 30:  # ignore tiny boxes
                continue
            cx,cy,bw,bh = bbox2yolo(x1,y1,x2,y2,W,H)
            if bw>0 and bh>0:
                lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # output names
        seq = img_path.parent.name
        out_stem = img_path.stem.replace("_leftImg8bit","")
        out_img = out_images / f"{seq}__{out_stem}.png"
        out_txt = out_labels / f"{seq}__{out_stem}.txt"

        shutil.copy2(img_path, out_img)
        out_txt.write_text("\n".join(lines), encoding="utf-8")

def make_valid_from_train(out_root, ratio=0.1, seed=42):
    random.seed(seed)
    train_imgs = sorted((out_root/"train/images").glob("*.png"))
    random.shuffle(train_imgs)
    n = int(len(train_imgs)*ratio)

    (out_root/"valid/images").mkdir(parents=True, exist_ok=True)
    (out_root/"valid/labels").mkdir(parents=True, exist_ok=True)

    for p in train_imgs[:n]:
        lbl = (out_root/"train/labels"/(p.stem+".txt"))
        shutil.move(str(p), str(out_root/"valid/images"/p.name))
        shutil.move(str(lbl), str(out_root/"valid/labels"/lbl.name))

def write_yaml(out_root):
    (out_root/"data.yaml").write_text(
f"""path: {out_root.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["obstacle"]
""", encoding="utf-8")

if __name__ == "__main__":
    LOST = Path("lost_and_found")
    OUT  = Path("lost_and_found_od_yolo")

    # convert train + test
    convert_split(LOST/"leftImg8bit/train", LOST/"gtCoarse", OUT/"train/images", OUT/"train/labels", "train")
    convert_split(LOST/"leftImg8bit/test",  LOST/"gtCoarse", OUT/"test/images",  OUT/"test/labels",  "test")

    # create valid from train (like your super dataset)
    make_valid_from_train(OUT, ratio=0.10)

    # yaml
    write_yaml(OUT)
    print("DONE:", OUT)
