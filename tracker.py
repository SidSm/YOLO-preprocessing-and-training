from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Load a model
model = YOLO("/home/sidsm/Projects/YOLO-preprocessing-and-training/datasets/dataset-2/runs/detect/train/weights/best.pt").to("cuda") # pretrained YOLO11n model

print(model.device)  # Should show 'cuda'

# Run batched inference on a list of images
results = model.track(
    #"rtsp://pco:VezeBlackTorch4@olsanska.blacktorch.eu:554/camera1", 
    #"https://www.youtube.com/watch?v=nQBOkGR_lg0",
    #"short-video1.mp4",
    "/home/sidsm/Projects/YOLO-preprocessing-and-training/test-clips/clip1-0.mp4",
    device=0,
    show=True,
    conf=0.5,
    iou=0.4
    #tracker="bytetrack.yaml"
)  # return a list of Results objects
