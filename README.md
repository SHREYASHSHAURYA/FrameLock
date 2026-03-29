# FrameLock 🔬

**Real-Time Surgical Video Motion Stabilization using Geometric Transformations**

FrameLock is a real-time video stabilization system designed for minimally invasive surgical procedures. It targets laparoscopic and endoscopic camera feeds, which are highly susceptible to hand tremors, fatigue-induced drift, and unintended positional inconsistencies. By applying continuous geometric corrections, FrameLock maintains a stable and consistent visual frame of reference — enhancing surgical precision and improving the reliability of downstream AI-based surgical analysis systems.

---

## Problem Statement

Endoscopic cameras used in laparoscopic surgeries are handheld and prone to motion artifacts including:

- **Translation** — horizontal and vertical jitter from hand movements
- **Rotation** — angular deviation from unintended camera roll
- **Scaling** — zoom variation due to changes in camera distance
- **Perspective distortion** — tilting and non-uniform distortion of the imaging plane

Such instability increases cognitive load on surgeons, reduces visual clarity during critical procedures, and degrades the performance of computer vision systems used for surgical tool tracking, tissue segmentation, and AI-assisted diagnostics.

---

## Features

- Real-time side-by-side comparison of original and stabilized video
- Region of Interest (ROI) detection using Laplacian variance — anatomy-aware, focuses on the most active surgical region
- Frame-to-frame motion estimation using optical flow and affine transformation
- Kalman filter-based trajectory smoothing for accurate, low-latency correction
- Support for all core geometric transformations — translation, rotation, scaling, affine, perspective, and reflection
- Edge artifact handling via border cropping and resizing
- Quantitative evaluation — centroid displacement metrics per video and averaged across dataset
- Visual evaluation — displacement plot showing raw vs stabilized motion over time
- Multi-video batch processing with per-video and aggregate metrics
- Interactive mode switching during playback via keyboard shortcuts

---

## Project Structure

```
FrameLock/
├── data/
│   ├── input/                  # Input video files (.mp4)
│   └── output/                 # Stabilized output videos
├── src/
│   ├── main.py                 # Main pipeline and processing loop
│   ├── video_io.py             # Video reading and display
│   ├── feature_detection.py    # Grayscale conversion and feature detection
│   ├── optical_flow.py         # Lucas-Kanade optical flow tracking
│   ├── motion_estimation.py    # Affine motion estimation
│   ├── smoothing.py            # Kalman filter trajectory smoothing
│   ├── roi.py                  # Anatomy-aware ROI detection
│   ├── transformations.py      # Geometric transformation matrices
│   └── evaluation.py           # Motion metrics and displacement plotting
├── README.md
└── .gitignore
```

---

## Installation

**Requirements:** Python 3.8+, OpenCV, NumPy

```bash
git clone https://github.com/SHREYASHSHAURYA/FrameLock
cd framelock
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
pip install opencv-python numpy
```

---

## Usage

Place input `.mp4` files in `data/input/`, then run:

```bash
python src/main.py
```

The pipeline processes each video, displays the real-time side-by-side comparison, and saves the stabilized output to `data/output/`. After each video, a displacement plot is shown comparing raw vs stabilized motion over time.

### Keyboard Controls

| Key | Mode                 |
| --- | -------------------- |
| `0` | Stabilized (default) |
| `1` | Translation demo     |
| `2` | Rotation demo        |
| `3` | Scaling demo         |
| `4` | Affine demo          |
| `5` | Perspective demo     |
| `6` | Reflection demo      |
| `q` | Quit current video   |

---

## How It Works

### 1. Feature Detection

Shi-Tomasi corner features are detected within the anatomy-aware ROI using `cv2.goodFeaturesToTrack`.

### 2. Optical Flow Tracking

Features are tracked frame-to-frame using the Lucas-Kanade pyramidal optical flow algorithm (`cv2.calcOpticalFlowPyrLK`) with a 3-level pyramid and 21×21 window.

### 3. Motion Estimation

An affine transformation matrix is estimated from tracked feature correspondences using `cv2.estimateAffine2D`, extracting translation (`dx`, `dy`) and rotation (`da`).

### 4. Trajectory Smoothing

A Kalman filter smooths the cumulative trajectory, separating intentional camera movement from unwanted shake. Only the correction component is applied to each frame.

### 5. Stabilization

The correction transform (translation + rotation) is applied via `cv2.warpAffine`. Edge artifacts from the warp are removed by cropping a 5% border and resizing back to original dimensions.

### 6. Anatomy-Aware ROI

The frame is divided into a 4×4 grid. Laplacian variance is computed per block every 15 frames to identify the sharpest, most textured region — the surgical site. The ROI smoothly tracks this region using exponential smoothing (α = 0.95).

---

## Evaluation

FrameLock measures **centroid displacement** — the mean Euclidean displacement per frame for both raw and stabilized motion:

```
Displacement = mean( sqrt(dx² + dy²) ) over all frames
Improvement  = Raw Displacement − Stabilized Displacement
```

Metrics are reported both globally (full frame) and for the ROI (surgical site), per video and averaged across the dataset. A displacement plot is generated after each video showing raw (red) vs stabilized (green) motion over time.

### Sample Results

| Video        | Raw Motion | Stabilized Motion | ROI Improvement |
| ------------ | ---------- | ----------------- | --------------- |
| normalshaky1 | 0.36       | 0.04              | 0.32            |
| surgical1    | 10.40      | 11.38             | 12.41           |
| surgical2    | 6.26       | 4.15              | 2.06            |
| surgical3    | 11.41      | 10.50             | 95.15           |
| surgical4    | 4.82       | 4.02              | 6.11            |
| **Average**  | **6.65**   | **6.02**          | **23.21**       |

> ROI improvement significantly exceeds global improvement — confirming the system correctly focuses stabilization on the surgical site while allowing intentional camera movement.

---

## Datasets

Tested on:

- Custom handheld shaky footage
- Laparoscopic surgical videos from [Pixabay](https://pixabay.com) (free, no registration required)

---

## Applications

- Laparoscopic and robotic surgery
- Tele-surgery and remote surgical guidance
- Computer-assisted diagnostics
- AI surgical tool tracking and tissue segmentation pipelines

---
