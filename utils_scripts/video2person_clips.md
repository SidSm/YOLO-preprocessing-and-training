# video2person.py - YOLO Video Person Detection and Clip Extraction

A Python utility that uses YOLOv8 to detect people in videos and automatically extract clips containing people.

## Description

This script processes a video file using YOLOv8 object detection to identify segments where people appear. It then uses FFmpeg to extract these segments as individual video clips. The tool is useful for automatically trimming videos to only include relevant sections with human presence, saving storage space and processing time for downstream applications.

## Features

- Detects people in videos using YOLOv8 object detection
- Samples frames at a configurable rate to balance speed and accuracy
- Merges nearby detections into continuous segments
- Adds customizable padding before and after each segment
- Extracts segments as separate video clips using FFmpeg
- Provides detailed progress information and validation checks

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Ultralytics YOLOv8 (`ultralytics`)
- FFmpeg (command-line tool)
- Standard libraries: `os`, `subprocess`, `time`

## Installation

1. Install the required Python packages:
   ```bash
   pip install opencv-python ultralytics
   ```

2. Ensure FFmpeg is installed on your system:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # On macOS with Homebrew
   brew install ffmpeg
   ```

3. Download a YOLOv8 model (the script uses 'yolo11m.pt')

## Configuration

The script uses the following parameters at the top of the file:

```python
VIDEO_PATH = "/path/to/input/video.mp4"       # Path to your input video file
OUTPUT_DIR = "/path/to/output/clips"          # Directory to store the extracted clips
SAMPLE_RATE = 0.5                             # Run inference on 0.5 frames per second
DETECTION_THRESHOLD = 0.5                     # Minimum confidence for valid detections
MAX_GAP = 3.0                                 # Maximum gap (seconds) to merge segments
PADDING = 2.0                                 # Seconds to pad at beginning and end
```

Adjust these parameters according to your needs before running the script.

## How It Works

1. The script analyzes the input video to get its properties (duration, FPS)
2. It runs a test clip extraction to verify FFmpeg functionality
3. It processes frames at the specified sampling rate using YOLOv8 for person detection
4. Detections are merged into segments based on the maximum allowed gap
5. Each segment is padded by the specified duration
6. FFmpeg is used to extract each segment as a separate clip without re-encoding

## Usage

1. Configure the parameters at the top of the script
2. Run the script:

```bash
python video2person_clips.py
```

3. When prompted with the video information, confirm by typing `Y`

## Output

- Video clips saved to the output directory with a naming pattern: `clip_1.mp4`, `clip_2.mp4`, etc.
- Console output with:
  - Video properties
  - Detection progress
  - Detected segments with timestamps
  - Extraction progress

## Performance Considerations

- The `SAMPLE_RATE` parameter significantly affects processing speed - lower values (more frames analyzed) increase accuracy but take longer
- Using a more powerful GPU will accelerate inference
- The script preserves the original video quality by using stream copying in FFmpeg
- For long videos, consider increasing the sample rate for faster processing

## Use Cases

- Automatically extracting relevant segments from surveillance footage
- Creating highlight reels from long videos
- Preprocessing videos for computer vision training datasets
- Removing empty sections from recordings

## Troubleshooting

- If you encounter permission errors when saving clips, try running the script with elevated privileges or check directory permissions
- For very large videos, you may need to increase system resources or process the video in chunks
- If detections are inconsistent, try adjusting the `DETECTION_THRESHOLD` and `SAMPLE_RATE` parameters

