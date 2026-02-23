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


def preprocess(X: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def run_kmeans(X_scaled):
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    distances = np.array([
        np.linalg.norm(X_scaled[i] - centroids[labels[i]])
        for i in range(len(X_scaled))
    ])
    return labels, distances, centroids

all_vec0, all_vec1 = [], []
vec0_path = '/home/user1/ariel/Sultan/data/feature_vec_0/individual_features/'
vec1_path = '/home/user1/ariel/Sultan/data/feature_vec_1/individual_features_10/'
# all_vec0([np.load(f) for f in random.sample(glob.glob(os.path.join(vec0_path, '*.npy')), k=10)])

all_vec0.extend([np.load(f) for f in random.sample(glob.glob(os.path.join(vec0_path, '*.npy')), k=10)])
all_vec1.extend([np.load(f) for f in random.sample(glob.glob(os.path.join(vec1_path, '*.npy')), k=5)])

# model resnet 50 for feature vector extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=True)

model = nn.Sequential(*list(resnet.children())[:-1])
model = model.to(device)
model.eval()

# Options: "webcam", "video", or "folder"
input_type = "folder" 

model_yv11 = YOLO("/home/user1/ariel/Sultan/yolov11/weights/best_L.pt")  
video_path = "path/to/your/video.mp4"
folder_path = "/home/user1/ariel/Sultan/data/rotem_mini/"

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
            # use this vector to apply to already existsing vectors in database to find the most similar one using kM
            X = np.vstack([all_vec0,all_vec1, f_vec])
            X_scaled = preprocess(X)
            labels, distances, centroids = run_kmeans(X_scaled)
            # add logics regarding what is the distance threshold to consider it as a match or not, and how to handle the labels
            # Calculate distance from test_vector to each centroid
            dist_to_centroid_0 = np.linalg.norm(f_vec - centroids[0])
            dist_to_centroid_1 = np.linalg.norm(f_vec - centroids[1])

            if dist_to_centroid_0 < dist_to_centroid_1:
                print("More similar to Class 0")
            else:
                print("More similar to Class 1")
            detection_data = {
                "image_id": img_name,
                "frame_index": frame_idx,
                "detection_index": box_idx,
                "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                "class": cls_id,
                "confidence": conf
            }
            
            current_frame_bboxes.append(detection_data)
            
            # Pinpointing the box in the console
            print(f"  -> Found Object {box_idx}: Class {cls_id} at {box.xyxy[0].cpu().numpy().astype(int).astype(int)}")
    counter += 1
    if counter==2:
        break


    cv2.destroyAllWindows()