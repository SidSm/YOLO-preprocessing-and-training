import cv2
import os

# ===========================
# CONFIGURATION PARAMETERS
# ===========================
OUTPUT_FRAMES = "/mnt/data-storage/frames/"#"datasets/olsanska/frames/"       # Path to the output video file
INPUT_CLIPS = "/mnt/data-storage/clips/" #"datasets/olsanska/clips/"          # Directory to source the extracted clips
SAMPLE_RATE = 15               # Sample and save 1 frame every 15 frames

# Get paths to all clips in input folder
# Count frame count per each clip and iterate over that
# Save to the output folder every x frame and do it for the rest of clips


file_paths = []
will_create_new_imgs = 0
for root, _, files in os.walk(INPUT_CLIPS):
    for file in files:
        path = os.path.join(root, file)
        file_paths.append(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error opening video file: {path}")
            exit(1)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_of_new_imgs = ((frame_count - 1) // SAMPLE_RATE) + 1 if frame_count > 0 else 0

        will_create_new_imgs += num_of_new_imgs

print(f"Around new {will_create_new_imgs} images will be created")
user_input = input("Are you okay to continue? Y/n")
if user_input != "Y":
    exit(0)
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
        cv2.imwrite(f"{OUTPUT_FRAMES}image_{im_number}-{clip_num}.jpg", frame)
        print(f"Created new image from clip {clip_num} at frame {current_frame_index}, img number: {im_number}")
        im_number = im_number + 1
        current_frame_index += SAMPLE_RATE
    clip_num = clip_num + 1
    cap.release()

print(f"------\nWork done. Created {im_number} new images from {clip_num} clips")