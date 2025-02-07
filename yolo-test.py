from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11m.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolo11n.engine")

# Run inference on a video
results = trt_model.predict(
    source="sample-30s.mp4",
    conf=0.5,
    iou=0.45,
    save=True,
    project="inference_results",
    name="tensorrt_run"
)
