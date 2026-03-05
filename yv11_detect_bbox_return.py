import cv2
import os
import glob
import random
from typing import List

import numpy as np
import torch
import joblib
from torch import nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  –  edit these to match your environment
# ─────────────────────────────────────────────────────────────────────────────
VEC0_PATH        = '/work/Sultan/data/rotem_non_pickups_f_vectors/individual_features/'
VEC1_PATH        = '/work/Sultan/data/rotem_pickups_f_vectors/individual_features/'

KMEANS_MODEL_PT  = '/work/Sultan/models/kmeans_model.pt'
SCALER_PKL       = '/work/Sultan/models/scaler.pkl'
RESNET_MODEL_PT  = '/work/Sultan/models/resnet_feature_model.pt'
YOLO_WEIGHTS     = '/work/Sultan/yolov11/weights/best_L.pt'

SAVE_DIR         = '/work/Sultan/runs/'
VIDEO_PATH       = 'path/to/your/video.mp4'
FOLDER_PATH      = '/work/Sultan/data/rotem_mini/'

# "webcam" | "video" | "folder"
INPUT_TYPE       = 'folder'

# Set to False to retrain KMeans + scaler from scratch
KM_EXISTS        = False

# Number of reference vectors sampled per class for training / label-verification
N_TRAIN_SAMPLES  = 24
YOLO_CONF        = 0.25
MIN_CROP_SIZE    = 10          # pixels; crops smaller than this are skipped
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# IMAGENET PREPROCESSING  –  must match what was used when the saved .npy
# feature vectors were produced.  Every crop goes through this before ResNet.
# ─────────────────────────────────────────────────────────────────────────────
imagenet_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_random_vectors(path: str, k: int) -> List[np.ndarray]:
    """Load k randomly sampled .npy feature vectors from *path*."""
    files = glob.glob(os.path.join(path, '*.npy'))
    if len(files) < k:
        raise ValueError(
            f"Requested {k} samples but only {len(files)} .npy files found in {path}"
        )
    return [np.load(f) for f in random.sample(files, k=k)]


def extract_feature_vector(crop_bgr: np.ndarray,
                            resnet_model: nn.Module,
                            device: torch.device) -> np.ndarray:
    """
    Convert a BGR crop (H×W×3 uint8) to a 2048-d ResNet-50 feature vector.

    Key fixes vs. original:
      • Resize to 224×224  (ResNet expects a fixed spatial size)
      • Apply ImageNet normalisation  (same distribution as training vectors)
      • No gradient computation needed at inference
    """
    tensor = imagenet_preprocess(crop_bgr)          # (3, 224, 224)  float32
    tensor = tensor.unsqueeze(0).to(device)          # (1, 3, 224, 224)
    with torch.no_grad():
        feat = resnet_model(tensor)                  # (1, 2048, 1, 1)
    return feat.squeeze().cpu().numpy()              # (2048,)


def save_image_with_bboxes(image_path: str,
                            bboxes: list,
                            title: str,
                            save_dir: str = SAVE_DIR,
                            color: tuple = (0, 255, 0),
                            thickness: int = 1) -> str | None:
    """Draw *bboxes* on the image at *image_path* and save to *save_dir*."""
    os.makedirs(save_dir, exist_ok=True)
    if not bboxes:
        return None

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Accept a single bbox as a flat list/tuple
    if isinstance(bboxes[0], (int, float)):
        bboxes = [bboxes]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_y    = max(y1 - 5, text_size[1])
        cv2.putText(img, title, (x1, text_y), font, font_scale, color, thickness)

    name, _   = os.path.splitext(os.path.basename(image_path))
    save_path = os.path.join(save_dir, f"{name}_bbox.jpg")
    cv2.imwrite(save_path, img)
    return save_path


def determine_cluster_labels(km: KMeans,
                              scaler: StandardScaler,
                              pickup_vecs: List[np.ndarray],
                              non_pickup_vecs: List[np.ndarray]) -> int:
    """
    KMeans assigns cluster indices (0 / 1) arbitrarily — they carry no
    semantic meaning on their own.  This function resolves the ambiguity by
    running known pickup AND non-pickup vectors through the trained model and
    using a majority vote to decide which integer label corresponds to 'pickup'.

    Strategy
    --------
    • Predict labels for up to 8 known pickup samples   → vote A
    • Predict labels for up to 8 known non-pickup samples → vote B
    • pickup_label  = the label that appears most in vote A
    • Sanity-check  : that same label should appear *least* in vote B
      (if it doesn't, print a warning — the clusters may overlap badly)

    Returns
    -------
    int  –  the KMeans cluster label (0 or 1) that represents 'pickup'.
    """
    n = min(8, len(pickup_vecs), len(non_pickup_vecs))

    # ── Pickup vote ───────────────────────────────────────────────────────
    pickup_sample  = np.vstack(pickup_vecs[:n])
    pickup_scaled  = scaler.transform(pickup_sample)
    pickup_preds   = km.predict(pickup_scaled)
    pickup_label   = int(np.bincount(pickup_preds).argmax())

    # ── Non-pickup vote ───────────────────────────────────────────────────
    non_pickup_sample  = np.vstack(non_pickup_vecs[:n])
    non_pickup_scaled  = scaler.transform(non_pickup_sample)
    non_pickup_preds   = km.predict(non_pickup_scaled)
    non_pickup_dominant = int(np.bincount(non_pickup_preds).argmax())

    # ── Sanity check ──────────────────────────────────────────────────────
    if pickup_label == non_pickup_dominant:
        print(
            f"[WARNING] Both pickup and non-pickup samples map mostly to "
            f"cluster {pickup_label}. The two classes may overlap in feature "
            f"space. Results may be unreliable."
        )
    else:
        print(
            f"[INFO] Cluster label verification passed — "
            f"pickup→{pickup_label}, non_pickup→{non_pickup_dominant}"
        )

    print(f"[INFO] PICKUP cluster label set to: {pickup_label}")
    return pickup_label


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load reference vectors for BOTH classes
#           (used for KMeans training and/or label verification)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading reference feature vectors …")
vec0_samples = load_random_vectors(VEC0_PATH, k=N_TRAIN_SAMPLES)  # common class : non-pickup
vec1_samples = load_random_vectors(VEC1_PATH, k=N_TRAIN_SAMPLES)  # rare class   : pickup

X_ref = np.vstack(vec0_samples + vec1_samples)   # (2*N, D)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Scaler + KMeans  (train once, persist; reload on subsequent runs)
#
#   FIX: The scaler is now saved alongside the KMeans model so that the
#        *exact* same mean/std is applied to every new crop vector.
#        Previously the scaler was re-fit on a fresh random sample every run,
#        meaning the coordinate system seen by KMeans.predict() shifted each
#        execution – causing all predictions to collapse to one cluster.
# ─────────────────────────────────────────────────────────────────────────────
if not KM_EXISTS:
    print("[INFO] Training KMeans + fitting scaler …")

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_ref)

    km = KMeans(n_clusters=2, random_state=42, n_init=24)
    km.fit(X_scaled)

    # Persist both objects together
    torch.save(km, KMEANS_MODEL_PT)
    joblib.dump(scaler, SCALER_PKL)
    print(f"[INFO] Saved KMeans  → {KMEANS_MODEL_PT}")
    print(f"[INFO] Saved scaler  → {SCALER_PKL}")

else:
    print("[INFO] Loading pre-trained KMeans + scaler …")
    km     = torch.load(KMEANS_MODEL_PT, weights_only=False)
    scaler = joblib.load(SCALER_PKL)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Determine which cluster label == pickup (rare class)
#
#   FIX: KMeans label assignment is non-deterministic.  We probe the model
#        with known pickup AND non-pickup vectors and use majority-vote to
#        discover the correct label, rather than hard-coding "cluster 1 == pickup".
#        A cross-check against non-pickup vectors also warns if the two classes
#        overlap badly in feature space.
# ─────────────────────────────────────────────────────────────────────────────
PICKUP_LABEL = determine_cluster_labels(km, scaler, vec1_samples, vec0_samples)
# Quick sanity check — print centroid distances
sample_non = np.vstack(vec0_samples[:5])
sample_pik = np.vstack(vec1_samples[:5])
print("non-pickup predictions:", km.predict(scaler.transform(sample_non)))
print("pickup    predictions:", km.predict(scaler.transform(sample_pik)))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Load ResNet-50 feature extractor
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
resnet_model = torch.load(
    RESNET_MODEL_PT,
    weights_only=False
).to(device).eval()
print(f"[INFO] Loaded ResNet feature extractor from {RESNET_MODEL_PT}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Load YOLOv11 detector
# ─────────────────────────────────────────────────────────────────────────────
model_yolo = YOLO(YOLO_WEIGHTS)

if   INPUT_TYPE == 'webcam': source = 0
elif INPUT_TYPE == 'video':  source = VIDEO_PATH
elif INPUT_TYPE == 'folder': source = FOLDER_PATH
else: raise ValueError("INPUT_TYPE must be 'webcam', 'video', or 'folder'.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Inference loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  Starting inference  |  source type : {INPUT_TYPE}")
print(f"  Pickup cluster label: {PICKUP_LABEL}")
print(f"{'─'*60}\n")

results = model_yolo.predict(source=source, stream=True, conf=YOLO_CONF)

for frame_idx, result in enumerate(results):

    img_name  = os.path.basename(result.path)
    img       = result.orig_img           # BGR numpy array  (H, W, 3)
    boxes     = result.boxes

    print(f"\n[Frame {frame_idx:04d}]  {img_name}  —  {len(boxes)} detection(s)")

    rare_bboxes          = []   # coordinates of pickup detections
    all_detection_data   = []   # full metadata for every detection

    for box_idx, box in enumerate(boxes):

        conf = float(box.conf[0])

        # ── Extract crop ──────────────────────────────────────────────────
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # FIX: Guard against degenerate / out-of-bounds crops
        x1 = max(x1, 0);  y1 = max(y1, 0)
        x2 = min(x2, img.shape[1]);  y2 = min(y2, img.shape[0])

        if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
            print(f"  [box {box_idx}] Skipping – crop too small ({x2-x1}×{y2-y1}px)")
            continue

        #crop = img[y1:y2, x1:x2]
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)   # BGR uint8

        # ── ResNet feature vector ─────────────────────────────────────────
        #   FIX: crop is now resized to 224×224 and ImageNet-normalised,
        #        matching the distribution of the saved .npy training vectors.
        f_vec = extract_feature_vector(crop, resnet_model, device)  # (2048,)

        # ── Scale with the PERSISTED scaler (same mean/std as training) ───
        f_vec_scaled = scaler.transform(f_vec.reshape(1, -1))        # (1, 2048)

        # ── KMeans prediction ─────────────────────────────────────────────
        cluster = int(km.predict(f_vec_scaled)[0])

        bbox_coords = [x1, y1, x2, y2]
        is_pickup   = (cluster == PICKUP_LABEL)
        cls_name    = 'pickup' if is_pickup else 'non_pickup'

        print(f"  [box {box_idx}] conf={conf:.2f}  cluster={cluster}  → {cls_name}  "
              f"bbox={bbox_coords}")

        if is_pickup:
            rare_bboxes.append(bbox_coords)

        all_detection_data.append({
            'image_id'        : img_name,
            'frame_index'     : frame_idx,
            'detection_index' : box_idx,
            'bbox'            : bbox_coords,
            'cluster'         : cluster,
            'class_name'      : cls_name,
            'confidence'      : conf,
        })

    # ── Save annotated image only if pickup detections were found ─────────
    if rare_bboxes:
        out_path = save_image_with_bboxes(
            result.path, rare_bboxes, 'pickup',
            save_dir=SAVE_DIR, color=(0, 255, 0), thickness=1
        )
        print(f"  [✓] Saved annotated image → {out_path}")
    else:
        print(f"  [–] No pickup detections in this frame.")
