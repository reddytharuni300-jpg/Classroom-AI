import os
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Config
IMG_DIR     = os.path.join("C:\\", "Users", "tharu", "Downloads", "dataset model", "output_jpg_images")
OUTPUT_DIR  = os.path.join("C:\\", "Users", "tharu", "OneDrive", "Desktop", "computer vision", "outputs", "visualizations")
SUMMARY_DIR = os.path.join("C:\\", "Users", "tharu", "OneDrive", "Desktop", "computer vision", "outputs")
YOLO_MODEL  = "yolov8m.pt"
CONF        = 0.35
IOU         = 0.45
IMG_SIZE    = 640
PERSON_CLS  = 0
MIN_AREA    = 0.001
MAX_AREA    = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

from ultralytics import YOLO
model = YOLO(YOLO_MODEL)
print("[OK] Loaded model: " + YOLO_MODEL)

img_paths = sorted(list(set(
    glob.glob(os.path.join(IMG_DIR, "*.jpg"))  +
    glob.glob(os.path.join(IMG_DIR, "*.png"))  +
    glob.glob(os.path.join(IMG_DIR, "*.jpeg")) +
    glob.glob(os.path.join(IMG_DIR, "**", "*.jpg"),  recursive=True) +
    glob.glob(os.path.join(IMG_DIR, "**", "*.png"),  recursive=True) +
    glob.glob(os.path.join(IMG_DIR, "**", "*.jpeg"), recursive=True)
)))
print("[OK] Found " + str(len(img_paths)) + " images")

records = []

for img_path in tqdm(img_paths, desc="Processing images"):
    results = model.predict(
        source=img_path,
        classes=[PERSON_CLS],
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    all_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0])
            all_boxes.append([x1, y1, x2, y2, c])

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        continue
    img_area = img_cv.shape[0] * img_cv.shape[1]
    boxes = [b for b in all_boxes
             if MIN_AREA <= ((b[2]-b[0])*(b[3]-b[1])) / img_area <= MAX_AREA]
    count = len(boxes)

    records.append({
        "image_id":   Path(img_path).stem,
        "image_path": img_path,
        "pred_count": count,
    })

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)

    for b in boxes:
        x1, y1, x2, y2, conf = b
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{conf:.2f}", color="lime",
                fontsize=7, fontweight="bold")

    ax.set_title(Path(img_path).stem + "  |  Students Detected: " + str(count),
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, Path(img_path).stem + "_detected.jpg")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

df = pd.DataFrame(records)
summary_path = os.path.join(SUMMARY_DIR, "all_images_counts.csv")
df[["image_id", "pred_count"]].to_csv(summary_path, index=False)

print("=" * 55)
print("Done! Processed " + str(len(df)) + " images")
print("Average count : " + str(round(df["pred_count"].mean(), 1)) + " students/image")
print("Max count     : " + str(df["pred_count"].max()) + " students")
print("Min count     : " + str(df["pred_count"].min()) + " students")
print("Visualizations: " + OUTPUT_DIR)
print("Summary CSV   : " + summary_path)
print("=" * 55)