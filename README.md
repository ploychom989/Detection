# 🎯 SIFT Visual SLAM

A real-time Visual SLAM (Simultaneous Localization and Mapping) system using SIFT feature detection and matching. Track camera movement and build trajectory maps in real-time using your webcam.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Real-time SIFT Feature Detection** - Detects up to 2000 keypoints per frame
- **FLANN-based Feature Matching** - Fast and accurate feature correspondence
- **Essential Matrix Estimation** - Robust camera pose recovery using RANSAC
- **Live Trajectory Visualization** - Real-time top-view camera path display
- **Headless Mode** - Record output without GUI for server environments

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy
```

### Run Live SLAM (with visualization)

```bash
python run_slam_live.py
```

This will:

- Open your webcam
- Display detected SIFT features
- Show real-time trajectory mapping
- Press `Q` to quit

### Run Headless Mode (no GUI)

```bash
python run_slam_headless.py
```

This will:

- Record 10 seconds of video
- Generate `output_video.avi` - Processed video with feature overlay
- Generate `trajectory.png` - Final camera trajectory map

## 📁 Project Structure

```
3DExclusive/
├── run_slam_live.py        # Live camera SLAM with GUI
├── run_slam_headless.py    # Headless SLAM for recording
├── feature_extraction.py   # SIFT feature extraction module
├── feature_matching.py     # Feature matching utilities
├── pose_estimation.py      # Camera pose estimation
├── 3d_reconstruction.py    # 3D point cloud reconstruction
├── test_camera.py          # Camera testing utility
├── settings.json           # Configuration file
└── output_video.avi        # Sample output video
```

## 🔧 How It Works

1. **Feature Extraction** - SIFT detects distinctive keypoints in each frame
2. **Feature Matching** - FLANN matcher finds correspondences between frames
3. **Essential Matrix** - Computed from matched points using 8-point algorithm + RANSAC
4. **Pose Recovery** - Decomposes Essential Matrix to get rotation and translation
5. **Trajectory Update** - Accumulates camera movement to build the path

## ⚙️ Camera Parameters

Default camera intrinsic matrix (modify in code for your camera):

```python
K = [[800,   0, 320],
     [  0, 800, 240],
     [  0,   0,   1]]
```

## 📊 Output Example

| Camera View            | Trajectory Map       |
| ---------------------- | -------------------- |
| Live keypoints overlay | Top-view camera path |

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Request features
- Submit pull requests

## 📝 License

This project is licensed under the MIT License.

---

Made with ❤️ for computer vision enthusiasts
