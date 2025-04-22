import os
import shutil
import random

INPUT_FRAMES = "/mnt/data-storage/frames/"#"datasets/olsanska/frames/"                      # Path to the input video file
OUT_RANDOM_FRAMES = "/mnt/data-storage/frame-random/" #"datasets/olsanska/clips/"           # Directory to source the extracted clips
FINAL_FRAME_COUNT = 800                                                                     # Number of frames that will generate
MIN_DISTANCE = 5                                                                            # A minimal distance in directory between these frames after sorting

file_paths = []
already_chosen_imgs_paths = []
already_chosen_indexes = []
frame_number = 0

def check_min_distance(numbers, min_distance):
    print("Checking minimal distances...")
    sorted_numbers = sorted(numbers)  # Sort numbers for sequential checking
    for i in range(1, len(sorted_numbers)):
        if sorted_numbers[i] - sorted_numbers[i - 1] < min_distance:
            return False, (sorted_numbers[i - 1], sorted_numbers[i])
    return True, None, None


for root, _, files in os.walk(INPUT_FRAMES):
    for file in files:
        path = os.path.join(root, file)
        file_paths.append(path)
        
        frame_number += 1
        
file_paths.sort() # Soft them by name so frames from each clip are behind each other

print(f"I found {frame_number} frames inside the folder, proceding...")
for i in range(0, FINAL_FRAME_COUNT):
    new_rand_index = 0
    while True:
        new_rand_index = random.randint(0, frame_number-1)
        if all(abs(new_rand_index - chosen) >= MIN_DISTANCE for chosen in already_chosen_indexes):
            already_chosen_indexes.append(new_rand_index)
            break  # Valid index found, exit loop


    shutil.copy2(file_paths[new_rand_index], OUT_RANDOM_FRAMES)
    #print(new_rand_index)
print(f"All new frames have been copied to {OUT_RANDOM_FRAMES}")


# Check if the condition holds for N=5
valid, violation1, violation2 = check_min_distance(already_chosen_indexes, MIN_DISTANCE)
    
if valid:
    print("All numbers are at least", MIN_DISTANCE, "positions apart.")
else:
    print("The condition is violated between:", violation1, violation2)