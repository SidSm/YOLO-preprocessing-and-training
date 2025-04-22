from ultralytics import YOLO
import torch
import cv2
import numpy as np

print(torch.version.cuda)
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print number of GPUs
print(torch.cuda.get_device_name(0))  # Should print your GPU name
print(torch.cuda.current_device())  # Should print 0 (default GPU)


# Load a model
model = YOLO("/home/sidsm/Projects/YOLO-preprocessing-and-training/datasets/dataset-2/runs/detect/train/weights/best.pt").to("cuda") # pretrained YOLO11n model
# Export the model to TensorRT

"""img = cv2.imread("/home/sidsm/Projects/YOLO-preprocessing-and-training/datasets/test/images/image_968-5305_jpg.rf.12f12590adc061796eaed6f720d429c1.jpg")  # Read image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Resize to a valid YOLO size (e.g., 640x640 or 1280x1280)
img = cv2.resize(img, (640, 640))  # Resize to 640x640 (divisible by 32)

# Convert to tensor and move to CUDA (FP16 for speed)
img = torch.from_numpy(img).to("cuda").half()

# Ensure correct tensor shape (batch, channels, height, width)
img = img.permute(2, 0, 1).unsqueeze(0)  # Rearrange dimensions"""

#model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
#trt_model = YOLO("yolo11n.engine")


print(model.device)  # Should show 'cuda'

# Run batched inference on a list of images
results = model.predict(
    #"rtsp://pco:VezeBlackTorch4@olsanska.blacktorch.eu:554/camera1", 
    #"https://www.youtube.com/watch?v=nQBOkGR_lg0",
    "/home/sidsm/Projects/YOLO-preprocessing-and-training/test-clips/clip3-0.mp4",
    imgsz=1280,
    conf=0.5, 
    verbose=True,
    device=0,
    stream=True,
    show=True,
    show_labels=False,
    half=True
)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    #result.save(filename="result-11.jpg")  # save to disk"""