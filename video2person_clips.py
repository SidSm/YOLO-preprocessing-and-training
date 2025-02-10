import cv2
import os
import subprocess
import time
from ultralytics import YOLO

# ===========================
# CONFIGURATION PARAMETERS
# ===========================
VIDEO_PATH = "/mnt/data-storage/Projects/yolo-training/video1.mp4"#"datasets/olsanska/short-video1.mp4"      # Path to your input video file
OUTPUT_DIR = "/mnt/data-storage/clips/" #"datasets/olsanska/clips/"#          # Directory to store the extracted clips
SAMPLE_RATE = 0.2               # Run inference on 0.2 frame per second (every 5 sec)
DETECTION_THRESHOLD = 0.5     # Minimum confidence required to consider a detection valid
MAX_GAP = 3.0                 # Maximum gap (in seconds) between detections to merge them into one segment
PADDING = 2.0                 # Seconds to pad at the beginning and end of each segment

# ===========================
# SET UP THE ENVIRONMENT
# ===========================
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the YOLOv8 model (this uses the ultralytics repo's pretrained model)
model = YOLO('yolo11n.pt')

# Open the video using OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file: {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(f"Video info: {VIDEO_PATH}")
print(f"  FPS: {fps:.2f}, Total Frames: {frame_count}, Duration: {duration:.2f} sec")
user_input = input("Do you want to continue? Press Y/N: ")
if user_input != "Y":
    print("Exiting...")
    exit(0)

# ===========================
# TEST SAVING GENERATED CLIP
# ===========================

output_file = os.path.join(OUTPUT_DIR, f"test_clip.mp4")
# Build the ffmpeg command:
# -ss sets the start time, -to sets the end time, and -c copy copies the stream without re-encoding.
command = [
    "ffmpeg",
    "-y",  # Overwrite output file if it exists
    "-i", VIDEO_PATH,
    "-ss", str(0),
    "-to", str(100),
    "-c", "copy",
    output_file
]
output = subprocess.run(command)
if int(output.returncode) == 1:
    print("Error: cannot save clip to the output destination folder, try running with sudo python -E")
    exit(0)
try:
    os.remove(output_file)
    print(f"File '{output_file}' deleted successfully.")
except FileNotFoundError:
    print(f"File '{output_file}' not found.")
except PermissionError:
    print(f"Permission denied: Unable to delete '{output_file}'.")
except Exception as e:
    print(f"Error deleting file: {e}")

# Since we want to run inference only once per second,
# we will skip approximately 'fps' frames between each detection.
skip_frames = int(fps / SAMPLE_RATE)

# ===========================
# RUN INFERENCE ON SAMPLED FRAMES
# ===========================
timestamps = []  # Will store times (in seconds) when a person is detected
current_frame_index = 0
start_time = time.time()

while current_frame_index < frame_count:
    # Set the video to the desired frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
    ret, frame = cap.read()
    if not ret:
        break

    current_time = current_frame_index / fps

    # Run detection on the current frame
    results = model(frame)
    person_detected = False

    # The YOLOv8 results contain a Boxes object.
    # Each detection is stored as [x1, y1, x2, y2, confidence, class]
    # We convert the detections to a list and check for class 0 ("person").
    boxes_data = results[0].boxes.data.tolist() if len(results[0].boxes.data) else []
    for box in boxes_data:
        conf = box[4]
        cls_id = int(box[5])
        # In YOLO models, class 0 is typically "person"
        if cls_id == 0 and conf >= DETECTION_THRESHOLD:
            person_detected = True
            break

    if person_detected:
        timestamps.append(current_time)
        print(f"Detection at {current_time:.2f} sec")

    # Jump ahead by skip_frames so that we process only 1 frame per second.
    current_frame_index += skip_frames

cap.release()

# ===========================
# MERGE TIMESTAMPS INTO SEGMENTS
# ===========================
if not timestamps:
    print("No person detected in the video.")
    exit(0)

segments = []
segment_start = timestamps[0]
segment_end = timestamps[0]

for t in timestamps[1:]:
    # If the gap between detections is less than or equal to MAX_GAP,
    # extend the current segment.
    if t - segment_end <= MAX_GAP:
        segment_end = t
    else:
        # Otherwise, close off the current segment and start a new one.
        start = max(0, segment_start - PADDING)
        end = min(duration, segment_end + PADDING)
        segments.append((start, end))
        segment_start = t
        segment_end = t

# Append the final segment.
segments.append((max(0, segment_start - PADDING), min(duration, segment_end + PADDING)))

print(f"\nDetected segments (with padding) in {time.time()-start_time} seconds:")
for idx, (start, end) in enumerate(segments, start=1):
    print(f"  Segment {idx}: {start:.2f} sec to {end:.2f} sec")

# ===========================
# EXTRACT CLIPS WITH FFMPEG
# ===========================
for i, (start, end) in enumerate(segments):
    output_file = os.path.join(OUTPUT_DIR, f"clip_{i+1}.mp4")
    # Build the ffmpeg command:
    # -ss sets the start time, -to sets the end time, and -c copy copies the stream without re-encoding.
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",  # Overwrite output file if it exists
        "-i", VIDEO_PATH,
        "-ss", str(start),
        "-to", str(end),
        "-c", "copy",
        output_file
    ]
    print(f"\nExtracting clip {i+1}/{segments.count()}: {output_file}")
    print(f"  From {start:.2f} sec to {end:.2f} sec")
    subprocess.run(command)
