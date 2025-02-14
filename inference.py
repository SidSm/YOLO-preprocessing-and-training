from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolo11m.pt")  # pretrained YOLO11n model
print(torch.cuda.is_available())  # Should be True
print(model.device)  # Should show 'cuda'

# Run batched inference on a list of images
results = model(
    "/mnt/data-storage/Projects/yolo-training/datasets/final-1-test/test/images/image_856-4708_jpg.rf.3808f34c4cff0cb06353a3ef7849438f.jpg", 
    imgsz=1280, 
    conf=0.5, 
    verbose=True
)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result-m-1280.jpg")  # save to disk