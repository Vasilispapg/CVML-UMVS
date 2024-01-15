
# Unsupervised Multi-Modal Video Summarization

## Overview
This project, Unsupervised Multi-Modal Video Summarization, automates the process of creating concise summaries from lengthy videos. It leverages state-of-the-art machine learning techniques, including object detection and unsupervised learning, without requiring pre-labeled training data.

## Objectives
- Summarize long videos efficiently.
- Utilize advanced object detection for content analysis.
- Apply unsupervised learning for automatic summarization.

## Implementation Details

### Data Extraction
- `audio.py`: Extracts audio features from video and processes them for further analysis.
- `visual.py`: Processes video frames for visual feature extraction, utilizing color features and optical flow.
- `objects.py`: Detects and encodes objects within video frames.
- `title.py`: Extracts and processes title features from the video's metadata.
- `getData.py`: General data retrieval interface for the other extraction modules.

### Feature Processing
- `clipImportance.py`: Calculates the importance of each video clip for summarization.
- `importances.py`: Defines functions for distance calculations and score normalization.
- `mapping.py`: Handles mapping between different data representations and formats.

### Knapsack Problem for Summarization
- `knapsack.py`: Implements the 0/1 knapsack algorithm to select the most important segments for video summarization.

### Video Creation
- `videoCreator.py`: Assembles the summarized video from selected frames.

### Main Process
- `__init__.py`: Orchestrates the summarization process, tying together data extraction, processing, and video creation.

## Expected Results
- Efficiency in summarization across video types.
- Effective summaries that encapsulate the core content.
- Enhanced user engagement with succinct summaries.

