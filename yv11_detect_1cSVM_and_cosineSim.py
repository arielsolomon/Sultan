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
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  –  edit these to match your environment
# ─────────────────────────────────────────────────────────────────────────────
VEC0_PATH       = '/work/Sultan/data/rotem_non_pickups_f_vectors/individual_features/'
VEC1_PATH       = '/work/Sultan/data/rotem_pickups_f_vectors/individual_features/'

OC_SVM_PKL      = '/work/Sultan/models/oc_svm.pkl'
SCALER_PKL      = '/work/Sultan/models/scaler.pkl'
CENTROIDS_NPZ   = '/work/Sultan/models/centroids.npz'   # stores both centroids
RESNET_MODEL_PT = '/work/Sultan/models/resnet_feature_model.pt'
YOLO_WEIGHTS    = '/work/Sultan/yolov11/weights/best_L.pt'

SAVE_DIR        = '/work/Sultan/runs/'
VIDEO_PATH      = 'path/to/your/video.mp4'
FOLDER_PATH     = '/work/Sultan/data/rotem_mini/'

# "webcam" | "video" | "folder"
INPUT_TYPE      = 'folder'

# Set to False to retrain One-Class SVM + scaler + centroids from scratch
MODELS_EXIST    = False

# How many .npy vectors to load per class
# Uses ALL available files if fewer than N_SAMPLES exist
N_SAMPLES       = 50

YOLO_CONF       = 0.25
MIN_CROP_SIZE   = 10        # px — crops smaller than this are skipped

# One-Class SVM sensitivity:
#   nu = upper bound on fraction of training outliers (0.01–0.2 typical)
#   Lower  → tighter boundary → fewer anomalies flagged (fewer false positives)
#   Higher → looser boundary  → more anomalies flagged  (fewer false negatives)
OC_SVM_NU       = 0.05
OC_SVM_GAMMA    = 'scale'   # 'scale' | 'auto' | float
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  –  identical to extract_f_vectors.py so inference vectors
#                   land in the same distribution as the saved .npy files.
# ─────────────────────────────────────────────────────────────────────────────
imagenet_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_vectors(path: str, k: int) -> np.ndarray:
    """
    Load up to k .npy feature vectors from *path*.
    Uses ALL available files if fewer than k exist.
    Returns ndarray of shape (n, D).
    """
    files = glob.glob(os.path.join(path, '*.npy'))
    if not files:
        raise ValueError(f"No .npy files found in {path}")
    if len(files) < k:
        print(f"[WARNING] Requested {k} samples but only {len(files)} found in "
              f"{os.path.basename(os.path.dirname(path))} — using all {len(files)}.")
        chosen = files
    else:
        chosen = random.sample(files, k=k)
    return np.vstack([np.load(f) for f in chosen])


def extract_feature_vector(crop_bgr: np.ndarray,
                            resnet_model: nn.Module,
                            device: torch.device) -> np.ndarray:
    """
    BGR crop (H×W×3 uint8)  →  2048-d ResNet-50 feature vector.

    Steps:
      1. BGR → RGB  (PIL assumes RGB; extract_f_vectors.py used PIL.open)
      2. Resize 256 + CenterCrop 224 + ImageNet normalise
         (identical to extract_f_vectors.py preprocessing)
      3. Forward pass, no grad
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)   # critical: match PIL
    tensor   = imagenet_preprocess(crop_rgb)                # (3, 224, 224)
    tensor   = tensor.unsqueeze(0).to(device)               # (1, 3, 224, 224)
    with torch.no_grad():
        feat = resnet_model(tensor)                         # (1, 2048, 1, 1)
    return feat.squeeze().cpu().numpy()                     # (2048,)


def compute_centroid(vecs_scaled: np.ndarray) -> np.ndarray:
    """L2-normalise each (scaled) vector then average → unit-norm centroid."""
    return normalize(vecs_scaled).mean(axis=0)              # (D,)


def cosine_to(f_vec: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine similarity between a single vector and a centroid."""
    return float(cosine_similarity(
        f_vec.reshape(1, -1),
        centroid.reshape(1, -1)
    )[0][0])


def save_image_with_bboxes(image_path: str,
                            bboxes: list,
                            label: str,
                            save_dir: str  = SAVE_DIR,
                            color: tuple   = (0, 255, 0),
                            thickness: int = 1) -> str | None:
    """Draw *bboxes* + *label* on the image at *image_path* and save."""
    os.makedirs(save_dir, exist_ok=True)
    if not bboxes:
        return None

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    if isinstance(bboxes[0], (int, float)):
        bboxes = [bboxes]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_y    = max(y1 - 5, text_size[1])
        cv2.putText(img, label, (x1, text_y), font, font_scale, color, thickness)

    name, _   = os.path.splitext(os.path.basename(image_path))
    save_path = os.path.join(save_dir, f"{name}_bbox.jpg")
    cv2.imwrite(save_path, img)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load reference vectors for both classes
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading reference feature vectors …")
X0 = load_vectors(VEC0_PATH, k=N_SAMPLES)   # non-pickup  (common / normal class)
X1 = load_vectors(VEC1_PATH, k=N_SAMPLES)   # pickup      (rare / anomaly class)
print(f"       non-pickup : {X0.shape}")
print(f"       pickup     : {X1.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Scaler  +  One-Class SVM  +  Centroids
#
# Architecture (Approach C):
#
#   GATE      : One-Class SVM trained ONLY on non-pickup vectors.
#               Detections inside the boundary → common car → discard.
#               Detections outside             → anomaly   → proceed to direction.
#
#   DIRECTION : Cosine similarity to each class centroid (computed in scaled space).
#               Whichever centroid the anomaly is closer to wins.
#               sim_pickup > sim_non_pickup  →  PICKUP  → save bbox
#               otherwise                   →  other anomaly (van/truck/moto) → discard
#
# Why centroids in scaled space:
#   StandardScaler makes every feature dimension contribute equally, preventing
#   a few high-variance ResNet channels from dominating the cosine similarity.
# ─────────────────────────────────────────────────────────────────────────────
if not MODELS_EXIST:
    print("[INFO] Training scaler + One-Class SVM + computing centroids …")

    # Fit scaler on ALL reference vectors (both classes) so it captures the
    # full dynamic range of the feature space, not just the common class.
    scaler = StandardScaler()
    scaler.fit(np.vstack([X0, X1]))

    X0_scaled = scaler.transform(X0)
    X1_scaled = scaler.transform(X1)

    # One-Class SVM — trained ONLY on non-pickup (the "normal" class)
    oc_svm = OneClassSVM(kernel='rbf', nu=OC_SVM_NU, gamma=OC_SVM_GAMMA)
    oc_svm.fit(X0_scaled)

    # Training-time sanity check
    pred0 = oc_svm.predict(X0_scaled)   # +1 = inlier, -1 = anomaly
    pred1 = oc_svm.predict(X1_scaled)
    print(f"  OC-SVM — non-pickup flagged as anomaly : "
          f"{(pred0==-1).mean()*100:.1f}%  (expect ~{OC_SVM_NU*100:.0f}%)")
    print(f"  OC-SVM — pickup    flagged as anomaly : "
          f"{(pred1==-1).mean()*100:.1f}%  (higher = better separation)")

    # Centroids computed in SCALED space
    centroid_non_pickup = compute_centroid(X0_scaled)
    centroid_pickup     = compute_centroid(X1_scaled)

    # Persist everything
    joblib.dump(scaler, SCALER_PKL)
    joblib.dump(oc_svm, OC_SVM_PKL)
    np.savez(CENTROIDS_NPZ,
             centroid_non_pickup=centroid_non_pickup,
             centroid_pickup=centroid_pickup)

    print(f"  Saved scaler    → {SCALER_PKL}")
    print(f"  Saved OC-SVM    → {OC_SVM_PKL}")
    print(f"  Saved centroids → {CENTROIDS_NPZ}")

else:
    print("[INFO] Loading pre-trained scaler + One-Class SVM + centroids …")
    scaler  = joblib.load(SCALER_PKL)
    oc_svm  = joblib.load(OC_SVM_PKL)
    data    = np.load(CENTROIDS_NPZ)
    centroid_non_pickup = data['centroid_non_pickup']
    centroid_pickup     = data['centroid_pickup']


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Sanity check on reference vectors
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Sanity check on reference vectors …")
n_check  = min(5, len(X0), len(X1))
X0_sc    = scaler.transform(X0[:n_check])
X1_sc    = scaler.transform(X1[:n_check])

svm_non  = oc_svm.predict(X0_sc)
svm_pick = oc_svm.predict(X1_sc)
print(f"  OC-SVM on non-pickup samples : {svm_non}   (expect mostly +1 / inlier)")
print(f"  OC-SVM on pickup    samples  : {svm_pick}  (expect mostly -1 / anomaly)")

print("  Centroid cosine check:")
for i in range(n_check):
    s0 = cosine_to(X0_sc[i], centroid_non_pickup)
    s1 = cosine_to(X0_sc[i], centroid_pickup)
    ok = '✓' if s0 >= s1 else '✗ mismatch'
    print(f"    non-pickup[{i}]  sim_non={s0:.3f}  sim_pick={s1:.3f}  {ok}")
for i in range(n_check):
    s0 = cosine_to(X1_sc[i], centroid_non_pickup)
    s1 = cosine_to(X1_sc[i], centroid_pickup)
    ok = '✓' if s1 > s0 else '✗ mismatch'
    print(f"    pickup    [{i}]  sim_non={s0:.3f}  sim_pick={s1:.3f}  {ok}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Load ResNet-50 feature extractor
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[INFO] Using device: {device}")

torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
resnet_model = torch.load(
    RESNET_MODEL_PT,
    weights_only=False
).to(device).eval()
print(f"[INFO] Loaded ResNet from {RESNET_MODEL_PT}")


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
#
# Per-detection decision flow:
#
#   crop → ResNet → scale
#       ↓
#   OC-SVM predict
#       ├── +1 (inlier)  → common car → no bbox saved
#       └── -1 (anomaly) → cosine to centroids
#                              ├── sim_pickup > sim_non  → PICKUP → save bbox
#                              └── sim_non   >= sim_pick → other anomaly → skip
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  Starting inference  |  source : {INPUT_TYPE}")
print(f"  OC-SVM  nu={OC_SVM_NU}  gamma={OC_SVM_GAMMA}")
print(f"{'─'*60}\n")

results = model_yolo.predict(source=source, stream=True, conf=YOLO_CONF)

for frame_idx, result in enumerate(results):

    img_name = os.path.basename(result.path)
    img      = result.orig_img       # BGR (H, W, 3)
    boxes    = result.boxes

    print(f"\n[Frame {frame_idx:04d}]  {img_name}  —  {len(boxes)} detection(s)")

    pickup_bboxes      = []
    all_detection_data = []

    for box_idx, box in enumerate(boxes):

        conf = float(box.conf[0])

        # ── Crop ──────────────────────────────────────────────────────────
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1 = max(x1, 0);  y1 = max(y1, 0)
        x2 = min(x2, img.shape[1]);  y2 = min(y2, img.shape[0])

        if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
            print(f"  [box {box_idx}] Skipping — crop too small ({x2-x1}×{y2-y1}px)")
            continue

        crop        = img[y1:y2, x1:x2]
        bbox_coords = [x1, y1, x2, y2]

        # ── ResNet feature vector ──────────────────────────────────────────
        f_vec        = extract_feature_vector(crop, resnet_model, device)  # (2048,)
        f_vec_scaled = scaler.transform(f_vec.reshape(1, -1))              # (1, 2048)

        # ── Cosine similarities (computed for all detections for logging) ──
        sim_non  = cosine_to(f_vec_scaled[0], centroid_non_pickup)
        sim_pick = cosine_to(f_vec_scaled[0], centroid_pickup)

        # ── One-Class SVM gate ─────────────────────────────────────────────
        svm_pred = oc_svm.predict(f_vec_scaled)[0]   # +1 inlier | -1 anomaly

        if svm_pred == 1:
            # Inside the common-car boundary → not a pickup
            cls_name = 'non_pickup'

        else:
            # Outside the common-car boundary → anomaly
            # Resolve direction via cosine similarity to class centroids
            if sim_pick > sim_non:
                cls_name = 'pickup'
                pickup_bboxes.append(bbox_coords)
            else:
                # Anomaly but geometrically closer to non-pickup centroid —
                # likely a van, truck, motorbike, or other unusual vehicle
                cls_name = 'other_anomaly'

        print(f"  [box {box_idx}] conf={conf:.2f}  "
              f"svm={'inlier' if svm_pred==1 else 'ANOMALY'}  "
              f"sim_non={sim_non:.3f}  sim_pick={sim_pick:.3f}  → {cls_name}")

        all_detection_data.append({
            'image_id'        : img_name,
            'frame_index'     : frame_idx,
            'detection_index' : box_idx,
            'bbox'            : bbox_coords,
            'class_name'      : cls_name,
            'svm_pred'        : int(svm_pred),
            'sim_non_pickup'  : sim_non,
            'sim_pickup'      : sim_pick,
            'confidence'      : conf,
        })

    # ── Save annotated image if any pickups were found ─────────────────────
    if pickup_bboxes:
        out_path = save_image_with_bboxes(
            result.path, pickup_bboxes, 'pickup',
            save_dir=SAVE_DIR, color=(0, 255, 0), thickness=1
        )
        print(f"  [✓] {len(pickup_bboxes)} pickup(s) — saved → {out_path}")
    else:
        print(f"  [–] No pickup detections in this frame.")
