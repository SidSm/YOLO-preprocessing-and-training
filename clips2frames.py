import cv2
import os
from ultralytics import YOLO

# ===========================
# CONFIGURATION PARAMETERS
# ===========================
OUTPUT_FRAMES = "datasets/olsanska/frames/"      # Path to the input video file
INPUT_CLIPS = "datasets/olsanska/clips/"          # Directory to source the extracted clips
SAMPLE_RATE = 5               # Save 1 frame every 5 frames

# Get paths to all clips in input folder
# Count frame count per each clip and iterate over that
# Save to the output folder every x frame and do it for the rest of clips


file_paths = []
for root, _, files in os.walk(INPUT_CLIPS):
    for file in files:
        file_paths.append(os.path.join(root, file))

clip_num = 0
im_number = 0

for file_path in file_paths:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error opening video file: {file_path}")
        exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print("------------------------")
    print(f"Video info: {file_path}")
    print(f"  FPS: {fps:.2f}, Total Frames: {frame_count}, Duration: {duration:.2f} sec")
    current_frame_index = 0
    while current_frame_index < frame_count:
        # Set the video to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{OUTPUT_FRAMES}image_{clip_num}-{im_number}.jpg", frame)
        print(f"Created new image from clip {clip_num} at frame {current_frame_index}, img number: {im_number}")
        im_number = im_number + 1
        current_frame_index += SAMPLE_RATE
    clip_num = clip_num + 1
    cap.release()

print(f"------\nWork done. Created {im_number+1} new images from {clip_num+1} clips")