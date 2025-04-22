# Random Frame Extractor

A Python script for randomly selecting frames from a source directory while maintaining a minimum distance between the selected frames.

## Description

This utility extracts a specified number of random frames from a source directory containing video frames. The script ensures that the selected frames maintain a minimum distance from each other (when sorted) to avoid selecting frames that are too similar or sequential.

## Features

- Random selection of frames from a source directory
- Enforces a minimum distance between selected frames
- Validates that all selected frames meet the minimum distance requirement
- Copies selected frames to an output directory

## Requirements

- Python 3.x
- Standard libraries: `os`, `shutil`, `random`

## Configuration

The script uses the following configuration parameters at the top of the file:

```python
INPUT_FRAMES = "/mnt/data-storage/frames/"      # Path to input frames directory
OUT_RANDOM_FRAMES = "/mnt/data-storage/frame-random/"  # Directory to store selected frames
FINAL_FRAME_COUNT = 800                         # Number of frames to extract
MIN_DISTANCE = 5                                # Minimum distance between frames
```

Adjust these parameters according to your needs before running the script.

## How It Works

1. The script scans the `INPUT_FRAMES` directory recursively to find all frames
2. Frames are sorted by name to ensure sequential order
3. The script randomly selects `FINAL_FRAME_COUNT` frames, ensuring each selection maintains at least `MIN_DISTANCE` from all previously selected frames
4. Selected frames are copied to the `OUT_RANDOM_FRAMES` directory
5. The script validates that all selected frames meet the minimum distance requirement

## Usage

1. Configure the parameters at the top of the script
2. Ensure the output directory exists
3. Run the script:

```bash
python random_pick_frames.py
```

## Output

- Selected frames are copied to the output directory
- Console output provides progress information and validation results

## Use Cases

- Creating datasets for machine learning from video sources
- Randomly sampling frames for video analysis
- Extracting diverse frames from long video sequences

## Notes

- The minimum distance parameter helps avoid selecting visually similar consecutive frames
- Increasing the minimum distance will result in more diverse frame selection
- The script sorts frames by filename, so proper sequential naming of frames is important