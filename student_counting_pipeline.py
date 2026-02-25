import os
import json
import glob
import math
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Dataset Paths ──────────────────────────────────────────────────────────
    DATA_DIR        = r"C:\Users\tharu\Downloads\dataset model\human_dtection_in_classroom"

    TRAIN_IMG_DIR   = r"C:\Users\tharu\Downloads\dataset model\human_dtection_in_classroom\train"
    VAL_IMG_DIR     = r"C:\Users\tharu\Downloads\dataset model\human_dtection_in_classroom\valid"
    TEST_IMG_DIR    = r"C:\Users\tharu\Downloads\dataset model\human_dtection_in_classroom\test"

    TRAIN_ANN_FILE  = r"C:\Users\tharu\Downloads\dataset model\human_dtection_in_classroom\train\_annotations.coco.json"
    VAL_ANN_FILE    = r"C:\Users\tharu\Downloads\dataset model\human_dtection_in_classroom\valid\_annotations.coco.json"

    RAW_IMG_DIR     = r"C:\Users\tharu\Downloads\dataset model\output_jpg_images"

    # ── Output Paths ───────────────────────────────────────────────────────────
    OUTPUT_DIR      = r"C:\Users\tharu\OneDrive\Desktop\computer vision\outputs"
    WEIGHTS_DIR     = r"C:\Users\tharu\OneDrive\Desktop\computer vision\weights"

    # ── YOLOv8 Settings ────────────────────────────────────────────────────────
    YOLO_MODEL      = "yolov8m.pt"
    PERSON_CLASS_ID = 0
    CONF_THRESHOLD  = 0.35
    IOU_THRESHOLD   = 0.45
    IMG_SIZE        = 640

    # ── CSRNet / Density Map Settings ──────────────────────────────────────────
    DENSITY_IMG_SIZE = (512, 512)
    SIGMA            = 15
    DENSITY_EPOCHS   = 50
    DENSITY_LR       = 1e-5
    DENSITY_BATCH    = 4

    # ── Hybrid Settings ────────────────────────────────────────────────────────
    HYBRID_DENSITY_THRESHOLD = 15

    # ── Device (set after instantiation) ───────────────────────────────────────
    DEVICE = "cpu"
    SEED   = 42

# FIX 1: Proper cfg instantiation outside the class
cfg = Config()
cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output dirs exist
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_coco_annotations(ann_file: str) -> Dict:
    """Load COCO-format annotation file."""
    with open(ann_file) as f:
        return json.load(f)


def get_person_count_from_coco(ann_data: Dict, image_id: int) -> int:
    """Count 'person' annotations for a given image_id."""
    person_cat_ids = {
        cat["id"] for cat in ann_data["categories"]
        if cat["name"].lower() in ("person", "student", "teacher", "instructor")
    }
    return sum(
        1 for ann in ann_data["annotations"]
        if ann["image_id"] == image_id and ann["category_id"] in person_cat_ids
    )


def compute_mae(preds: List[float], gts: List[float]) -> float:
    return float(np.mean(np.abs(np.array(preds) - np.array(gts))))


def compute_rmse(preds: List[float], gts: List[float]) -> float:
    return float(np.sqrt(np.mean((np.array(preds) - np.array(gts)) ** 2)))


def compute_mape(preds: List[float], gts: List[float], eps=1e-6) -> float:
    gts_arr = np.array(gts, dtype=float)
    return float(np.mean(np.abs((np.array(preds) - gts_arr) / (gts_arr + eps))) * 100)


def evaluate_counts(preds: List[float], gts: List[float]) -> Dict:
    return {
        "MAE":   round(compute_mae(preds, gts),  3),
        "RMSE":  round(compute_rmse(preds, gts), 3),
        "MAPE%": round(compute_mape(preds, gts), 2),
        "N":     len(preds),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. APPROACH 1 — DETECTION-BASED COUNTING (YOLOv8)
# ──────────────────────────────────────────────────────────────────────────────

class DetectionCounter:
    """
    Uses a pretrained / fine-tuned YOLOv8 model to detect persons
    and derives the count from the number of bounding boxes.
    """

    def __init__(self, model_path: str = cfg.YOLO_MODEL,
                 conf: float = cfg.CONF_THRESHOLD,
                 iou: float = cfg.IOU_THRESHOLD):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf  = conf
        self.iou   = iou
        print(f"[DetectionCounter] Loaded model: {model_path}")

    # ------------------------------------------------------------------
    def predict_image(self, img_path: str) -> Tuple[int, List[List[float]]]:
        results = self.model.predict(
            source=img_path,
            classes=[cfg.PERSON_CLASS_ID],
            conf=self.conf,
            iou=self.iou,
            imgsz=cfg.IMG_SIZE,
            verbose=False,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append([x1, y1, x2, y2, conf])
        return len(boxes), boxes

    # ------------------------------------------------------------------
    def predict_with_area_filter(
        self,
        img_path: str,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.5,
    ) -> Tuple[int, List[List[float]]]:
        img = cv2.imread(img_path)
        if img is None:
            return 0, []
        img_area = img.shape[0] * img.shape[1]
        _, boxes = self.predict_image(img_path)

        filtered = []
        for b in boxes:
            x1, y1, x2, y2, conf = b
            box_area = (x2 - x1) * (y2 - y1)
            ratio    = box_area / img_area
            if min_area_ratio <= ratio <= max_area_ratio:
                filtered.append(b)

        return len(filtered), filtered

    # ------------------------------------------------------------------
    # FIX 2: predict_folder searches recursively into subdirectories
    # FIX 3: guard against empty DataFrame
    # ------------------------------------------------------------------
    def predict_folder(
        self,
        img_dir: str,
        save_json: bool = True,
        output_path: str = None,
    ) -> pd.DataFrame:
        # Recursive image search
        img_paths = sorted(list(set(
            glob.glob(os.path.join(img_dir, "*.jpg"))                        +
            glob.glob(os.path.join(img_dir, "*.png"))                        +
            glob.glob(os.path.join(img_dir, "*.jpeg"))                       +
            glob.glob(os.path.join(img_dir, "**/*.jpg"),  recursive=True)    +
            glob.glob(os.path.join(img_dir, "**/*.png"),  recursive=True)    +
            glob.glob(os.path.join(img_dir, "**/*.jpeg"), recursive=True)
        )))
        print(f"  Found {len(img_paths)} images in: {img_dir}")

        # FIX 3: return early with empty DataFrame if no images found
        if len(img_paths) == 0:
            print("  ⚠️  No images found — returning empty DataFrame")
            return pd.DataFrame(columns=["image_id", "image_path", "pred_count", "boxes"])

        records = []
        for p in tqdm(img_paths, desc="Detection counting"):
            count, boxes = self.predict_with_area_filter(p)
            records.append({
                "image_id":   Path(p).stem,
                "image_path": p,
                "pred_count": count,
                "boxes":      boxes,
            })
        df = pd.DataFrame(records)

        if save_json and len(df) > 0:
            out = output_path or os.path.join(cfg.OUTPUT_DIR, "detection_counts.json")
            df[["image_id", "pred_count"]].to_json(out, orient="records", indent=2)
            print(f"[DetectionCounter] Saved predictions → {out}")

        return df

    # ------------------------------------------------------------------
    def visualise(self, img_path: str, save_path: str = None):
        count, boxes = self.predict_with_area_filter(img_path)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
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
        ax.set_title(f"Detection Count: {count}", fontsize=16)
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualisation → {save_path}")
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4. APPROACH 2 — DENSITY MAP REGRESSION (CSRNet-lite)
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def generate_density_map(
    img_h: int, img_w: int,
    keypoints: List[Tuple[float, float]],
    sigma: int = cfg.SIGMA,
) -> np.ndarray:
    density = np.zeros((img_h, img_w), dtype=np.float32)
    k_size  = 6 * sigma + 1
    kernel  = gaussian_kernel(k_size, sigma)
    pad     = k_size // 2

    for (x, y) in keypoints:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < img_w and 0 <= yi < img_h:
            r0, r1 = yi - pad, yi + pad + 1
            c0, c1 = xi - pad, xi + pad + 1
            kr0 = max(0, -r0);  kr1 = k_size - max(0, r1 - img_h)
            kc0 = max(0, -c0);  kc1 = k_size - max(0, c1 - img_w)
            r0, r1 = max(0, r0), min(img_h, r1)
            c0, c1 = max(0, c0), min(img_w, c1)
            if r1 > r0 and c1 > c0:
                density[r0:r1, c0:c1] += kernel[kr0:kr1, kc0:kc1]

    return density


def bboxes_to_keypoints(bboxes: List[List[float]]) -> List[Tuple[float, float]]:
    return [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in bboxes]


class DensityMapDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: Tuple[int, int] = cfg.DENSITY_IMG_SIZE,
        augment: bool = True,
    ):
        self.img_dir  = img_dir
        self.ann      = load_coco_annotations(ann_file)
        self.img_size = img_size
        self.augment  = augment

        person_cat_ids = {
            c["id"] for c in self.ann["categories"]
            if c["name"].lower() in ("person", "student", "teacher", "instructor")
        }
        self.id2info: Dict[int, Dict] = {}
        for img in self.ann["images"]:
            self.id2info[img["id"]] = {"file_name": img["file_name"], "bboxes": []}
        for ann in self.ann["annotations"]:
            if ann["category_id"] in person_cat_ids:
                iid = ann["image_id"]
                if iid in self.id2info:
                    self.id2info[iid]["bboxes"].append(ann["bbox"])

        self.ids = list(self.id2info.keys())

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        info   = self.id2info[self.ids[idx]]
        img_p  = os.path.join(self.img_dir, info["file_name"])
        img    = Image.open(img_p).convert("RGB")
        ow, oh = img.size

        img_r  = img.resize(self.img_size, Image.BILINEAR)
        sw, sh = self.img_size[0] / ow, self.img_size[1] / oh

        kps = [
            (cx * sw, cy * sh)
            for (cx, cy) in bboxes_to_keypoints(info["bboxes"])
        ]

        dm_h, dm_w = self.img_size[1] // 8, self.img_size[0] // 8
        kps_down   = [(x / 8, y / 8) for (x, y) in kps]
        density    = generate_density_map(dm_h, dm_w, kps_down,
                                          sigma=max(1, cfg.SIGMA // 8))

        if self.augment and np.random.rand() > 0.5:
            img_r   = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            density = density[:, ::-1].copy()

        img_t     = self.transform(img_r)
        density_t = torch.from_numpy(density).unsqueeze(0)
        count     = torch.tensor(float(density_t.sum()), dtype=torch.float32)

        return img_t, density_t, count


class CSRNetLite(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),  nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  3, padding=2, dilation=2),  nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.relu(x)
        return x


class DensityTrainer:
    def __init__(self, model: nn.Module = None):
        self.model = model or CSRNetLite(pretrained=True)
        self.model = self.model.to(cfg.DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.DENSITY_LR, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        self.loss_fn = nn.MSELoss()

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, densities, _ in loader:
            imgs, densities = imgs.to(cfg.DEVICE), densities.to(cfg.DEVICE)
            preds = self.model(imgs)
            loss  = self.loss_fn(preds, densities)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        mae_sum = rmse_sum = loss_sum = 0.0
        for imgs, densities, counts in loader:
            imgs, densities = imgs.to(cfg.DEVICE), densities.to(cfg.DEVICE)
            preds = self.model(imgs)
            loss_sum += self.loss_fn(preds, densities).item()
            pred_counts = preds.sum(dim=(1, 2, 3)).cpu()
            true_counts = counts
            mae_sum  += float(torch.abs(pred_counts - true_counts).mean())
            rmse_sum += float(((pred_counts - true_counts) ** 2).mean().sqrt())
        n = len(loader)
        return loss_sum / n, mae_sum / n, rmse_sum / n

    def fit(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        epochs: int = cfg.DENSITY_EPOCHS,
        save_best: bool = True,
    ):
        train_dl = DataLoader(train_ds, batch_size=cfg.DENSITY_BATCH,
                              shuffle=True,  num_workers=2, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=cfg.DENSITY_BATCH,
                              shuffle=False, num_workers=2, pin_memory=True)

        best_mae  = float("inf")
        history   = {"train_loss": [], "val_mae": [], "val_rmse": []}
        best_path = os.path.join(cfg.WEIGHTS_DIR, "csrnet_best.pt")

        for epoch in range(1, epochs + 1):
            t_loss                 = self.train_epoch(train_dl)
            v_loss, v_mae, v_rmse  = self.eval_epoch(val_dl)
            self.scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_mae"].append(v_mae)
            history["val_rmse"].append(v_rmse)

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {t_loss:.4f} | "
                  f"Val MAE: {v_mae:.2f} | Val RMSE: {v_rmse:.2f}")

            if save_best and v_mae < best_mae:
                best_mae = v_mae
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✅ Best model saved (MAE={best_mae:.2f})")

        return history

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Saved weights → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
        print(f"Loaded weights ← {path}")


class DensityCounter:
    def __init__(self, model_path: str):
        self.model = CSRNetLite(pretrained=False).to(cfg.DEVICE)
        self.model.load_state_dict(
            torch.load(model_path, map_location=cfg.DEVICE)
        )
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(cfg.DENSITY_IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        print(f"[DensityCounter] Loaded model: {model_path}")

    @torch.no_grad()
    def predict_image(self, img_path: str) -> Tuple[float, np.ndarray]:
        img    = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(cfg.DEVICE)
        pred   = self.model(tensor)
        dm     = pred.squeeze().cpu().numpy()
        count  = float(dm.sum())
        return count, dm

    def predict_folder(self, img_dir: str, save_json: bool = True) -> pd.DataFrame:
        img_paths = sorted(list(set(
            glob.glob(os.path.join(img_dir, "*.jpg"))                        +
            glob.glob(os.path.join(img_dir, "*.png"))                        +
            glob.glob(os.path.join(img_dir, "*.jpeg"))                       +
            glob.glob(os.path.join(img_dir, "**/*.jpg"),  recursive=True)    +
            glob.glob(os.path.join(img_dir, "**/*.png"),  recursive=True)    +
            glob.glob(os.path.join(img_dir, "**/*.jpeg"), recursive=True)
        )))
        print(f"  Found {len(img_paths)} images in: {img_dir}")

        if len(img_paths) == 0:
            print("  ⚠️  No images found — returning empty DataFrame")
            return pd.DataFrame(columns=["image_id", "image_path", "pred_count"])

        records = []
        for p in tqdm(img_paths, desc="Density counting"):
            count, _ = self.predict_image(p)
            records.append({
                "image_id":   Path(p).stem,
                "image_path": p,
                "pred_count": round(count),
            })
        df = pd.DataFrame(records)
        if save_json and len(df) > 0:
            out = os.path.join(cfg.OUTPUT_DIR, "density_counts.json")
            df[["image_id", "pred_count"]].to_json(out, orient="records", indent=2)
        return df

    def visualise_density(self, img_path: str, save_path: str = None):
        img   = Image.open(img_path).convert("RGB")
        count, dm = self.predict_image(img_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(dm, cmap="jet")
        axes[1].set_title(f"Density Map  |  Est. Count: {count:.1f}", fontsize=14)
        axes[1].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. APPROACH 3 — HYBRID ADAPTIVE COUNTER
# ──────────────────────────────────────────────────────────────────────────────

class HybridCounter:
    def __init__(
        self,
        det_counter: DetectionCounter,
        density_counter: DensityCounter,
        threshold: int = cfg.HYBRID_DENSITY_THRESHOLD,
        blend_alpha: float = 0.4,
    ):
        self.det     = det_counter
        self.density = density_counter
        self.thr     = threshold
        self.alpha   = blend_alpha

    def predict_image(self, img_path: str) -> Tuple[int, str]:
        det_count, _ = self.det.predict_with_area_filter(img_path)

        if det_count < self.thr:
            return det_count, "detection"

        density_count, _ = self.density.predict_image(img_path)
        blended = (1 - self.alpha) * det_count + self.alpha * density_count
        return int(round(blended)), "hybrid"

    def predict_folder(self, img_dir: str, save_json: bool = True) -> pd.DataFrame:
        img_paths = sorted(list(set(
            glob.glob(os.path.join(img_dir, "*.jpg"))                        +
            glob.glob(os.path.join(img_dir, "*.png"))                        +
            glob.glob(os.path.join(img_dir, "*.jpeg"))                       +
            glob.glob(os.path.join(img_dir, "**/*.jpg"),  recursive=True)    +
            glob.glob(os.path.join(img_dir, "**/*.png"),  recursive=True)    +
            glob.glob(os.path.join(img_dir, "**/*.jpeg"), recursive=True)
        )))
        print(f"  Found {len(img_paths)} images in: {img_dir}")

        if len(img_paths) == 0:
            print("  ⚠️  No images found — returning empty DataFrame")
            return pd.DataFrame(columns=["image_id", "image_path", "pred_count", "method"])

        records = []
        for p in tqdm(img_paths, desc="Hybrid counting"):
            count, method = self.predict_image(p)
            records.append({
                "image_id":   Path(p).stem,
                "image_path": p,
                "pred_count": count,
                "method":     method,
            })
        df = pd.DataFrame(records)
        if save_json and len(df) > 0:
            out = os.path.join(cfg.OUTPUT_DIR, "hybrid_counts.json")
            df[["image_id", "pred_count"]].to_json(out, orient="records", indent=2)
            print(f"[HybridCounter] Saved predictions → {out}")
        return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION & REPORTING
# ──────────────────────────────────────────────────────────────────────────────

class CountingEvaluator:
    def __init__(self, ann_file: str):
        self.ann = load_coco_annotations(ann_file)
        person_cat_ids = {
            c["id"] for c in self.ann["categories"]
            if c["name"].lower() in ("person", "student", "teacher", "instructor")
        }
        self.gt: Dict[str, int] = {}
        id2stem = {img["id"]: Path(img["file_name"]).stem
                   for img in self.ann["images"]}
        counts: Dict[str, int] = {s: 0 for s in id2stem.values()}
        for ann in self.ann["annotations"]:
            if ann["category_id"] in person_cat_ids:
                stem = id2stem.get(ann["image_id"])
                if stem:
                    counts[stem] += 1
        self.gt = counts

    def evaluate_df(self, df: pd.DataFrame, label: str = "Model") -> Dict:
        if len(df) == 0:
            print(f"⚠️  evaluate_df: empty DataFrame for '{label}' — skipping")
            return {}
        merged = df.set_index("image_id")["pred_count"]
        preds, gts = [], []
        for img_id, gt_count in self.gt.items():
            if img_id in merged.index:
                preds.append(float(merged[img_id]))
                gts.append(float(gt_count))

        if not preds:
            print(f"⚠️  No matching image_ids between predictions and ground truth for '{label}'")
            return {}

        metrics = evaluate_counts(preds, gts)
        print(f"\n{'='*50}")
        print(f"  {label} Evaluation")
        print(f"{'='*50}")
        for k, v in metrics.items():
            print(f"  {k:8s}: {v}")
        print(f"{'='*50}")
        return metrics

    def plot_error_distribution(
        self,
        df: pd.DataFrame,
        label: str = "Model",
        save_path: str = None,
    ):
        if len(df) == 0:
            print(f"⚠️  plot_error_distribution: empty DataFrame — skipping")
            return
        merged = df.set_index("image_id")["pred_count"]
        errors = []
        for img_id, gt_count in self.gt.items():
            if img_id in merged.index:
                errors.append(float(merged[img_id]) - gt_count)

        if not errors:
            return

        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=30, color="steelblue", edgecolor="white")
        plt.axvline(0, color="red", linewidth=2, linestyle="--", label="Perfect")
        plt.xlabel("Prediction Error (pred − gt)")
        plt.ylabel("Frequency")
        plt.title(f"{label} — Error Distribution  (MAE={np.mean(np.abs(errors)):.2f})")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def compare_methods(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for name, df in results.items():
            if len(df) == 0:
                continue
            merged = df.set_index("image_id")["pred_count"]
            preds, gts = [], []
            for img_id, gt in self.gt.items():
                if img_id in merged.index:
                    preds.append(float(merged[img_id]))
                    gts.append(float(gt))
            m = evaluate_counts(preds, gts)
            m["Method"] = name
            rows.append(m)

        comp = pd.DataFrame(rows).set_index("Method").sort_values("MAE")
        print("\n── Method Comparison ──────────────────────────")
        print(comp.to_string())
        print("───────────────────────────────────────────────")
        return comp


# ──────────────────────────────────────────────────────────────────────────────
# 7. TEST-SET SUBMISSION GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_submission(
    predictions_df: pd.DataFrame,
    output_file: str = None,
) -> pd.DataFrame:
    if len(predictions_df) == 0:
        print("⚠️  generate_submission: empty DataFrame — nothing to save")
        return predictions_df

    sub = predictions_df[["image_id", "pred_count"]].copy()
    sub.columns = ["image_id", "student_count"]
    sub["student_count"] = sub["student_count"].round().astype(int).clip(lower=0)

    out = output_file or os.path.join(cfg.OUTPUT_DIR, "submission.csv")
    sub.to_csv(out, index=False)
    print(f"Submission saved → {out}")
    print(sub.head())
    return sub


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_detection_pipeline():
    """End-to-end detection-based counting pipeline."""
    print("\n🔍 Running Detection-Based Counting Pipeline")
    print("─" * 50)

    counter = DetectionCounter()

    # FIX 4: find a valid image directory (val → train → raw)
    img_dir = None
    for candidate in [cfg.VAL_IMG_DIR, cfg.TRAIN_IMG_DIR, cfg.RAW_IMG_DIR]:
        if os.path.exists(candidate):
            found = (
                glob.glob(os.path.join(candidate, "*.jpg")) +
                glob.glob(os.path.join(candidate, "**/*.jpg"), recursive=True)
            )
            if found:
                img_dir = candidate
                print(f"  Using image dir: {img_dir}  ({len(found)} images found)")
                break

    if img_dir is None:
        print("❌ No images found in any configured directory. Check your Config paths.")
        return pd.DataFrame(columns=["image_id", "pred_count"])

    val_df = counter.predict_folder(img_dir)

    if len(val_df) == 0:
        print("❌ predict_folder returned empty — no images were processed")
        return val_df

    # Evaluate if annotations are available
    if img_dir == cfg.VAL_IMG_DIR and os.path.exists(cfg.VAL_ANN_FILE):
        evaluator = CountingEvaluator(cfg.VAL_ANN_FILE)
        evaluator.evaluate_df(val_df, label="YOLOv8 Detection Counter")
        evaluator.plot_error_distribution(
            val_df, label="YOLOv8",
            save_path=os.path.join(cfg.OUTPUT_DIR, "detection_error_dist.png")
        )

    # FIX 5: guard generate_submission against missing test dir
    if os.path.exists(cfg.TEST_IMG_DIR):
        test_df = counter.predict_folder(cfg.TEST_IMG_DIR, save_json=True)
        if len(test_df) > 0:
            generate_submission(test_df)
        else:
            print("⚠️  Test folder empty — generating submission from val predictions")
            generate_submission(val_df)
    else:
        print(f"⚠️  TEST_IMG_DIR not found: {cfg.TEST_IMG_DIR}")
        print("   Generating submission from val predictions instead...")
        generate_submission(val_df)

    return val_df


def run_density_pipeline(train: bool = True):
    """Train CSRNet-lite and run density-based counting."""
    print("\n🌡️  Running Density Map Pipeline")
    print("─" * 50)

    trainer = DensityTrainer()
    best_path = os.path.join(cfg.WEIGHTS_DIR, "csrnet_best.pt")

    if train or not os.path.exists(best_path):
        train_ds = DensityMapDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_ANN_FILE, augment=True)
        val_ds   = DensityMapDataset(cfg.VAL_IMG_DIR,   cfg.VAL_ANN_FILE,   augment=False)
        history  = trainer.fit(train_ds, val_ds, epochs=cfg.DENSITY_EPOCHS)
    else:
        trainer.load(best_path)

    counter = DensityCounter(best_path)
    test_df = counter.predict_folder(cfg.TEST_IMG_DIR)
    generate_submission(test_df, output_file=os.path.join(cfg.OUTPUT_DIR, "submission_density.csv"))
    return test_df


def run_comparison(det_df: pd.DataFrame, density_df: pd.DataFrame):
    """Compare detection vs density counting on validation data."""
    if not os.path.exists(cfg.VAL_ANN_FILE):
        print("No validation annotations found — skipping comparison.")
        return

    evaluator = CountingEvaluator(cfg.VAL_ANN_FILE)
    evaluator.compare_methods({
        "YOLOv8 Detection": det_df,
        "CSRNet Density":   density_df,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 9. QUICK DEMO — single image, no dataset needed
# ──────────────────────────────────────────────────────────────────────────────

def demo_single_image(img_path: str):
    print(f"\n📷 Demo: {img_path}")
    counter = DetectionCounter()
    count, boxes = counter.predict_with_area_filter(img_path)
    print(f"   Detected persons: {count}")
    for i, b in enumerate(boxes, 1):
        print(f"   Box {i:2d}: x1={b[0]:.0f} y1={b[1]:.0f} "
              f"x2={b[2]:.0f} y2={b[3]:.0f} conf={b[4]:.2f}")
    counter.visualise(
        img_path,
        save_path=os.path.join(cfg.OUTPUT_DIR, "demo_visualisation.png")
    )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Counting Pipeline")
    parser.add_argument("--mode",
        choices=["detection", "density", "hybrid", "demo"],
        default="detection",
        help="Which pipeline to run"
    )
    parser.add_argument("--img", type=str, default=None,
        help="Path to single image (used with --mode demo)")
    parser.add_argument("--train-density", action="store_true",
        help="Train CSRNet from scratch (used with --mode density)")
    args = parser.parse_args()

    if args.mode == "demo" and args.img:
        demo_single_image(args.img)

    elif args.mode == "detection":
        run_detection_pipeline()

    elif args.mode == "density":
        run_density_pipeline(train=args.train_density)

    elif args.mode == "hybrid":
        det_counter = DetectionCounter()
        density_counter = DensityCounter(
            os.path.join(cfg.WEIGHTS_DIR, "csrnet_best.pt")
        )
        hybrid = HybridCounter(det_counter, density_counter)
        test_df = hybrid.predict_folder(cfg.TEST_IMG_DIR)
        generate_submission(test_df)

    print("Pipeline execution completed.")