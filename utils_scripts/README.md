# Video Processing Toolkit

This repository contains a collection of Python utilities for video processing, frame extraction, and computer vision applications. These tools work together to provide a complete pipeline for processing video content - from detecting people in videos using YOLO, to extracting frames at regular intervals, and selecting diverse random frames for dataset creation.

Each utility in this toolkit is designed to address a specific part of the video processing workflow while maintaining compatibility with the others. Whether you're creating training datasets for machine learning, analyzing video content, or extracting specific segments from longer recordings, these scripts provide the building blocks for efficient video data processing.

## Tools Overview

The toolkit includes the following utilities:

1. **video2person_clips.py - YOLO Video Person Detection** - Automatically detect people in videos and extract relevant clips
2. **clips2frames.py - Video Frame Extractor** - Extract frames from videos at regular intervals
3. **random_pick_frames.py - Random Frame Selector** - Select diverse frames from a collection while maintaining minimum distance

---



