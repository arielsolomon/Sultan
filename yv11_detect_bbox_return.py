import cv2
import os
from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn as nn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import glob
import random
from typing import Union, List, Tuple


def preprocess(X: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def run_kmeans(X_scaled, km=None, train=False):
    if train:
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(X_scaled)
        return km
    else:
        labels = km.fit_predict(X_scaled)
        centroids = km.cluster_centers_
        distances = np.array([
            np.linalg.norm(X_scaled[i] - centroids[labels[i]])
            for i in range(len(X_scaled))
            ])
    return labels, distances, centroids

def save_image_with_bboxes(image_path,
                           bboxes,
                           title,
                           save_dir ="/work/Sultan/runs/",
                           color = (0,255,0),
                           thickness = 1):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    if isinstance(bboxes[0], (int,float)):
        bboxes = [bboxes]
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_x = x1
        text_y = max(y1 - 5, text_size[1])
        cv2.putText(img, title, (text_x, text_y), font, font_scale, color, thickness)
    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)
    save_path = os.path.join(save_dir, f"{name}_bbox.jpg")
    cv2.imwrite(save_path, img)
    return save_path

all_vec0, all_vec1 = [], []
vec0_path = '/work/Sultan/data/feature_vec_0/individual_features/'
vec1_path = '/work/Sultan/data/feature_vec_1/individual_features_all/'
# all_vec0([np.load(f) for f in random.sample(glob.glob(os.path.join(vec0_path, '*.npy')), k=10)])

all_vec0.extend([np.load(f) for f in random.sample(glob.glob(os.path.join(vec0_path, '*.npy')), k=25)])
all_vec1.extend([np.load(f) for f in random.sample(glob.glob(os.path.join(vec1_path, '*.npy')), k=25)])
X = np.vstack([all_vec0,all_vec1])
X_scaled = preprocess(X)
# do kmeans train only once
km = run_kmeans(X_scaled, train=True)
# model resnet 50 for feature vector extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# resnet = models.resnet50(pretrained=True)

# model = nn.Sequential(*list(resnet.children())[:-1])  
model = torch.load("/work/Sultan/models/resnet_feature_model.pt").to(device).eval()            

# Options: "webcam", "video", or "folder"
input_type = "folder" 

model_yv11 = YOLO("/work/Sultan/yv11/weights/yolo11l.pt")  
video_path = "path/to/your/video.mp4"
folder_path = "/work/Sultan/data/nopickup/images/all_classes/test_yolo/"

# Determine the source string/index
if input_type == "webcam":
    source = 0
elif input_type == "video":
    source = video_path
elif input_type == "folder":
    source = folder_path
else:
    raise ValueError("Invalid input_type. Choose 'webcam', 'video', or 'folder'.")

results = model_yv11.predict(source=source, stream=True, conf=0.25)

print(f"--- Starting Inference on {input_type} ---")

# 3. DISCRIMINATION: The Frame-by-Frame Loop
counter = 0
for frame_idx, result in enumerate(results):
    
    # Identify the current image/frame
    # For folders, this is the filename. For video, it's the frame sequence.
    img_name = os.path.basename(result.path)
    print(f"\nProcessing Index [{frame_idx}] | Source: {img_name}")
    
    # result.boxes contains ALL detections for THIS image only
    boxes = result.boxes
    img = result.orig_img
    # List to store coordinates for your "Further Use"
    current_frame_bboxes = []
    boxes1 = []
    for box_idx, box in enumerate(boxes):
        conf = float(box.conf[0])
        if conf > 0.25:
            # Extract coordinates in [x1, y1, x2, y2] format
            # .cpu().numpy() converts from GPU tensor to standard array
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop = img[y1:y2, x1:x2]
            # Meta-data
            cls_id = int(box.cls[0])
            f_vec = model(torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float().to(device)).squeeze().cpu().detach().numpy()
            f_vec_scaled = preprocess(f_vec.reshape(1, -1)) # Scale the new vector using the same scaler
            cluster = km.predict(f_vec_scaled)[0]#

            if cluster == 0:
                print("More similar to Class 0")

            else:
                print("More similar to Class 1")
                boxes1.append(box.xyxy[0].cpu().numpy().astype(int).tolist()) 

            detection_data = {
                "image_id": img_name,
                "frame_index": frame_idx,
                "detection_index": box_idx,
                "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                "class": cluster,
                "confidence": conf
            }
            
            current_frame_bboxes.append(detection_data)
            
            # Pinpointing the box in the console
            print(f"  -> Found Object {box_idx}: Class {cluster} at {box.xyxy[0].cpu().numpy().astype(int).astype(int)}")
    save_image_with_bboxes(result.path, boxes1, 'pickup', "/work/Sultan/runs/", (0,255,0), 1)
    counter += 1
    if counter==6:
        break


    cv2.destroyAllWindows()
