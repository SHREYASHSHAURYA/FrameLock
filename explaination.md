# FrameLock — Technical Explanation

> A deep-dive into the architecture, algorithms, design decisions, and data flow of the FrameLock surgical video stabilization system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Pipeline](#3-core-pipeline)
   - [3.1 Feature Detection](#31-feature-detection)
   - [3.2 Optical Flow Tracking](#32-optical-flow-tracking)
   - [3.3 Motion Estimation](#33-motion-estimation)
   - [3.4 Trajectory Smoothing](#34-trajectory-smoothing)
   - [3.5 Stabilization and Border Correction](#35-stabilization-and-border-correction)
   - [3.6 Anatomy-Aware ROI](#36-anatomy-aware-roi)
4. [Geometric Transformations](#4-geometric-transformations)
5. [Evaluation and Metrics](#5-evaluation-and-metrics)
6. [Web API Layer](#6-web-api-layer)
7. [React Dashboard](#7-react-dashboard)
8. [Data Flow](#8-data-flow)
9. [Design Decisions](#9-design-decisions)
10. [Limitations and Future Work](#10-limitations-and-future-work)

---

## 1. Project Overview

FrameLock is a real-time video stabilization system built specifically for minimally invasive surgical footage — laparoscopic and endoscopic camera feeds. These cameras are handheld during procedures and are highly susceptible to motion artifacts: hand tremor, fatigue-induced drift, unintentional rotation, and perspective distortion. Left uncorrected, these artifacts increase cognitive load on surgeons and degrade the performance of downstream computer vision systems used for tool tracking, tissue segmentation, and AI-assisted diagnostics.

FrameLock solves this by continuously estimating the inter-frame motion using optical flow and feature tracking, smoothing the cumulative camera trajectory with a Kalman filter, and applying the inverse correction transform to each frame — producing a stabilized output in real time.

The system has two interfaces:

- **Terminal mode** (`main.py`) — processes all videos in `data/input/` sequentially with an OpenCV display window
- **Web dashboard** (`api.py` + `ui/`) — a browser-based interface with live MJPEG streams, real-time metric graphs, and full pipeline control

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│         data/input/*.mp4   or   Webcam (device 0)          │
└────────────────────────────┬────────────────────────────────┘
                             │ frames
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE PIPELINE                          │
│                                                             │
│  ROI Detection → Feature Detection → Optical Flow          │
│       → Motion Estimation → Kalman Smoothing               │
│       → Correction Transform → Stabilized Frame            │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐     ┌────────────────────────────────┐
│   TERMINAL OUTPUT    │     │         WEB LAYER              │
│                      │     │                                │
│  OpenCV window       │     │  api.py (Flask + Waitress)     │
│  Displacement plots  │     │  ├── MJPEG stream (raw)        │
│  Console metrics     │     │  ├── MJPEG stream (stabilized) │
│  Output .mp4 writer  │     │  ├── SSE metrics stream        │
└──────────────────────┘     │  └── REST endpoints            │
                             │                                │
                             │  ui/ (React + Vite)            │
                             │  ├── Live video feeds          │
                             │  ├── Real-time graphs          │
                             │  ├── HUD metrics               │
                             │  └── Batch results table       │
                             └────────────────────────────────┘
```

The pipeline is entirely frame-by-frame. No future frames are read ahead — every stabilization decision is made using only the current and previous frame, which is what enables real-time operation.

---

## 3. Core Pipeline

### 3.1 Feature Detection

**File:** `src/feature_detection.py`

Each frame is first converted to grayscale. Feature detection uses the **Shi-Tomasi corner detector** (`cv2.goodFeaturesToTrack`), which identifies points with strong gradients in both x and y directions — corners and textured regions. These are more stable across frames than edges or blobs.

Detection is run:

- On the first frame, within the anatomy-aware ROI
- Any time the tracked point count drops below a threshold (adaptive recovery)

The detector is confined to the ROI patch rather than the full frame, which significantly reduces noise from background regions (trocars, draping, instrument handles) that move independently of the surgical site.

---

### 3.2 Optical Flow Tracking

**File:** `src/optical_flow.py`

Detected features are tracked from the previous frame to the current frame using **Lucas-Kanade pyramidal optical flow** (`cv2.calcOpticalFlowPyrLK`).

Key parameters:

- **3-level image pyramid** — allows tracking of both fine and coarse motion
- **21×21 search window** — large enough to handle moderate inter-frame displacement
- **Forward-backward consistency check** — tracks points forward then backward; points with high round-trip error are discarded as unreliable

Only points that pass the consistency check are passed to motion estimation. This ensures that the motion model is not corrupted by points that slipped off a surface, became occluded, or moved independently (e.g. surgical instruments crossing the ROI).

---

### 3.3 Motion Estimation

**File:** `src/motion_estimation.py`

From the filtered set of matched point pairs `(prev_pts, curr_pts)`, an **affine transformation matrix** is estimated using `cv2.estimateAffine2D` with RANSAC.

RANSAC (Random Sample Consensus) iteratively fits the affine model to random subsets of point pairs and selects the model with the most inliers — making the estimate robust to outliers from independently moving instruments or reflections.

The affine matrix encodes:

- **Translation** — `dx`, `dy` (horizontal and vertical shift)
- **Rotation** — `da` (angular deviation in radians)
- **Scaling** — captured implicitly in the affine coefficients

These components are extracted separately and accumulated into a cumulative trajectory.

---

### 3.4 Trajectory Smoothing

**File:** `src/smoothing.py`

Raw inter-frame motion is noisy. Simply inverting it would produce jittery correction. FrameLock separates **intentional camera movement** (panning, zooming in on tissue) from **unintentional shake** (tremor, breathing, fatigue) by smoothing the cumulative trajectory.

A **Kalman filter** is used with:

- **State vector:** cumulative `[x, y, angle]`
- **Process noise:** models the fact that intentional movement can change the trajectory smoothly
- **Measurement noise:** models the uncertainty in each frame's motion estimate

The Kalman filter produces a smooth estimate of where the camera _should_ be. The difference between the raw cumulative position and the smoothed position is the correction that needs to be applied to each frame:

```
correction = smoothed_position[t] - raw_position[t]
```

This correction is intentionally _not_ a full inverse — it preserves slow, deliberate camera movements while cancelling high-frequency shake.

---

### 3.5 Stabilization and Border Correction

**File:** `src/transformations.py`, `src/main.py` / `src/api.py`

The correction transform (translation + rotation) is applied to the current frame using `cv2.warpAffine`. This warps the frame to cancel the detected shake.

Warping introduces **black border artifacts** at the edges of the frame — regions that fall outside the original frame boundary after the transform. These are removed by:

1. Cropping a **5% border** on all sides
2. Resizing the cropped result back to the original frame dimensions

This introduces a slight zoom-in effect (~5% on each edge) but produces a clean, artifact-free output with no black regions.

---

### 3.6 Anatomy-Aware ROI

**File:** `src/roi.py`

Rather than tracking features across the entire frame, FrameLock identifies the **most anatomically informative region** and concentrates stabilization there. This is important because surgical frames contain large uniform regions (draping, organ surfaces) and only a small textured area (the actual surgical site with tissue texture, sutures, or instrument tips).

**Algorithm:**

1. The frame is divided into a **4×4 grid** of 16 equal blocks
2. Every 15 frames, the **Laplacian variance** of each block is computed — a measure of local sharpness and texture
3. The block with the highest variance is selected as the surgical site
4. The ROI centre is updated using **exponential smoothing** (α = 0.95) to prevent sudden jumps:

```
cx = α × prev_cx + (1 - α) × new_cx
cy = α × prev_cy + (1 - α) × new_cy
```

5. The ROI is expanded to a fixed-size bounding box around this centre

This means the ROI smoothly tracks the surgical site as the camera repositions, without recomputing every frame. The ROI rectangle is drawn on the stabilized output feed so the operator can see exactly what region is being prioritised.

---

## 4. Geometric Transformations

**File:** `src/transformations.py`

FrameLock implements all geometric corrections as explicit **2×3 affine matrices** applied via `cv2.warpAffine`. This makes the transformation pipeline modular, composable, and easy to extend.

| Transform   | Matrix Form                               | Use                                |
| ----------- | ----------------------------------------- | ---------------------------------- |
| Translation | `[[1, 0, dx], [0, 1, dy]]`                | Corrects horizontal/vertical drift |
| Rotation    | `[[cos θ, -sin θ, 0], [sin θ, cos θ, 0]]` | Corrects angular shake             |
| Scaling     | `[[s, 0, 0], [0, s, 0]]`                  | Zoom correction                    |
| Affine      | Combination of the above                  | General correction                 |
| Perspective | 3×3 homography via `cv2.warpPerspective`  | Non-planar distortion              |
| Reflection  | `cv2.flip`                                | Horizontal mirror                  |

For rotation and scaling, transforms are applied **about the image centre** rather than the origin. This is achieved by composing three matrices: translate to origin → apply transform → translate back:

```
M_centred = T2 × M × T1
where T1 shifts to origin, T2 shifts back
```

The demo modes (keyboard 0–6 in terminal, mode buttons in dashboard) apply exaggerated versions of each transform to the raw frame, making the effect of each transformation type visually clear for educational and demonstration purposes.

---

## 5. Evaluation and Metrics

**Files:** `src/evaluation.py`, `src/advanced_metrics.py`

### Primary Metric: Centroid Displacement

For each frame, the Euclidean displacement is computed from the estimated motion:

```
displacement = sqrt(dx² + dy²)
```

The **running mean displacement** is tracked separately for:

- **Raw** — the unstabilized motion
- **Stabilized** — the residual motion after correction

### ROI Displacement

The same metric is computed using only feature points that fall within the anatomy-aware ROI. This gives a more clinically relevant measure — how much the _surgical site_ is moving, rather than the whole frame.

### Improvement

```
Improvement (%) = (ROI_raw - ROI_stab) / ROI_raw × 100
```

A positive value means shake at the surgical site has been reduced. ROI improvement consistently exceeds global improvement because the system is specifically designed to prioritise the surgical site.

### Advanced Metrics (`advanced_metrics.py`)

- Mean, standard deviation, and maximum displacement for both raw and stabilized
- Per-component analysis: `dX` mean and `dY` mean separately
- Motion intensity timeline — per-frame displacement bar chart
- Regional heatmap — spatial distribution of motion across frame regions
- Multi-panel analytics plots: time series, component plots, phase space, histogram

---

## 6. Web API Layer

**File:** `src/api.py`

The Flask API acts as the bridge between the Python stabilization pipeline and the browser-based dashboard. It runs the pipeline in a **background thread**, keeping the main thread free to serve HTTP requests without blocking.

### Threading Model

```
Main thread:   Flask/Waitress HTTP server
               ├── serves MJPEG frames
               ├── serves SSE events
               └── handles REST requests

Background thread:  _pipeline()
               ├── reads frames from video/camera
               ├── runs full stabilization pipeline
               ├── writes JPEG bytes to shared state
               └── pushes SSE metric events
```

Shared state between threads is protected by two locks:

- `frame_lock` — guards the raw and stabilized JPEG byte buffers
- `lock` — guards the SSE event queue

### MJPEG Streaming

Each video feed is served as a **multipart HTTP response** (`multipart/x-mixed-replace`). The generator function polls the shared frame buffer and yields a new JPEG boundary whenever a new frame is available:

```python
def _mjpeg_generator(key):
    last_jpg = None
    while True:
        jpg = processing_state.get(key)
        if jpg is not None and jpg is not last_jpg:
            last_jpg = jpg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        else:
            time.sleep(0.001)
```

The `last_jpg is not last_jpg` identity check (not equality) ensures the generator only yields when the pipeline has actually produced a new frame, avoiding duplicate frame transmission and CPU waste from busy-polling.

### SSE Metrics Stream

Per-frame metrics are pushed as **Server-Sent Events** to the browser. The pipeline emits:

- `start` — signals the beginning of a new video with total frame count and FPS
- `metrics` — per-frame displacement, ROI scores, FPS, feature count (throttled to every 5 frames)
- `summary` — final statistics after processing completes
- `done` — signals clean termination
- `error` — propagates pipeline exceptions to the UI

### Server

The API uses **Waitress** as the WSGI server instead of Flask's built-in development server. Flask's dev server is single-threaded by default and cannot serve two concurrent MJPEG streams (raw + stabilized) alongside SSE and REST requests simultaneously. Waitress provides a production-grade multi-threaded server with 8 worker threads, sufficient for all concurrent stream consumers.

---

## 7. React Dashboard

**File:** `ui/src/App.jsx`

The dashboard is a single-page React application built with Vite. It communicates with the API exclusively through:

- `fetch()` for REST calls (start, stop, mode change)
- `EventSource` for the SSE metrics stream
- `<img src="...">` pointing at the MJPEG stream URLs — browsers natively decode multipart JPEG streams in image elements

### State Management

All application state lives in the root `App` component and is passed down as props. The key pieces of state are:

| State       | Type    | Purpose                                      |
| ----------- | ------- | -------------------------------------------- |
| `running`   | boolean | Whether the pipeline is active               |
| `source`    | string  | Current video filename or `"camera"`         |
| `metrics`   | object  | Latest frame metrics from SSE                |
| `summaries` | object  | Per-video summary results, keyed by filename |
| `mode`      | string  | Current transform mode                       |
| `hist`      | ref     | Rolling history arrays for sparkline graphs  |

History arrays are stored in a `useRef` (not `useState`) to avoid triggering re-renders on every frame — only the graph components read from them, and they update on their own render cycle.

### Video Feed Component

`VideoFeed` manages a single MJPEG stream. It has two display modes:

- **Inline** — fixed 16:9 aspect ratio container with the stream image filling it via `objectFit: contain`
- **Fullscreen** — `position: fixed, inset: 0, z-index: 9999` overlay with the image letterboxed inside, closed by clicking the backdrop or pressing Escape

Each feed tracks its own `imgError` state to gracefully handle stream unavailability.

### Graphs

Six sparkline panels update in real time as SSE metrics arrive:

| Panel                  | Data                           |
| ---------------------- | ------------------------------ |
| Displacement Magnitude | Raw vs stabilized running mean |
| ROI Displacement       | ROI raw vs ROI stabilized      |
| X Component            | Absolute dX over time          |
| Y Component            | Absolute dY over time          |
| FPS Timeline           | Pipeline throughput            |
| Motion Intensity       | Per-frame heatbar              |

Sparklines are rendered as SVG `<polyline>` elements. All values are normalised to the current maximum in the history window, so the scale adjusts automatically as motion intensity changes.

---

## 8. Data Flow

A complete trace of one frame through the system:

```
1. cap.read() → raw BGR frame

2. convert_to_grayscale(frame) → curr_gray

3. track_features(prev_gray, curr_gray, prev_points)
   → (prev_pts, curr_pts)  [filtered by consistency check]

4. estimate_motion(prev_pts, curr_pts)
   → dx, dy, da  [RANSAC affine estimation]

5. trajectory.update(dx, dy, da)
   smoothed = trajectory.smooth_kalman()
   → sx, sy, sa  [smoothed cumulative position]

6. diff_x = (sx - prev_sx) - dx   [correction = smoothed delta - raw delta]
   diff_y = (sy - prev_sy) - dy
   diff_a = (sa - prev_sa) - da

7. combined_M = rotation(diff_a) + translation(diff_x, diff_y)
   stabilized = warpAffine(frame, combined_M)

8. crop 5% border → resize to original dimensions

9. apply demo mode (if not "final") → output frame

10. draw ROI rectangle on output frame

11. imencode(raw_frame) → raw JPEG bytes  ─┐
    imencode(output_frame) → stab JPEG bytes ─┤→ stored in shared state
                                              │   → served by MJPEG generators

12. push SSE metrics event (every 5 frames) ──→ received by browser EventSource
                                                 → updates React state
                                                 → re-renders HUD + graphs

13. write hstack(frame, output) → output .mp4
```

---

## 9. Design Decisions

### Why Kalman filter over simple moving average?

A moving average introduces a fixed lag proportional to the window size — the correction always lags behind the actual motion. The Kalman filter is a recursive estimator that maintains an optimal balance between the predicted state (based on the model) and the measured state (based on optical flow), producing a smoother result with lower lag.

### Why ROI-focused feature detection?

Detecting features across the full frame in surgical video is problematic. Large uniform regions (abdominal wall, draping) produce few stable features. Instruments move independently and would corrupt the camera motion estimate. The ROI approach concentrates tracking on the one region that matters — the surgical site — producing a more accurate motion estimate with fewer computational resources.

### Why MJPEG over WebRTC or HLS?

MJPEG is the simplest possible streaming protocol for this use case. It requires no additional libraries, no codec negotiation, and works natively in browser `<img>` elements. The trade-off is higher bandwidth than H.264, but on localhost this is irrelevant. WebRTC would add significant complexity for synchronisation and signalling with no benefit in a single-client local deployment.

### Why Waitress over Flask dev server?

Two simultaneous MJPEG streams plus SSE plus REST requests require genuine concurrency. Flask's dev server handles requests sequentially in a single thread — one MJPEG stream would block all other requests. Waitress provides true multi-threaded request handling with minimal configuration.

### Why SSE over WebSocket for metrics?

Metrics flow in one direction only — server to client. SSE is sufficient for unidirectional streaming and is simpler to implement and debug than WebSocket. The `EventSource` API in the browser handles reconnection automatically.

---

## 10. Limitations and Future Work

### Current Limitations

**No temporal lookahead** — the Kalman filter is causal (uses only past frames). Offline processing could use a two-pass approach — first pass to collect the full trajectory, second pass to apply optimally smoothed corrections — producing better results on pre-recorded video.

**Single-threaded pipeline** — feature detection, optical flow, motion estimation, and JPEG encoding all run sequentially in one thread. For high-resolution video, this limits throughput.

**Fixed ROI grid** — the 4×4 grid and 15-frame update interval are hardcoded. Adaptive grid density based on frame content would improve ROI localisation in diverse surgical scenarios.

**No depth awareness** — the system treats the surgical scene as a 2D plane. Depth changes (camera moving toward or away from tissue) manifest as scaling artifacts that the affine model handles imperfectly.

**MJPEG bandwidth** — at full resolution, two simultaneous MJPEG streams consume significant bandwidth. This is not an issue on localhost but would be prohibitive over a network.

### Potential Improvements

- **Offline two-pass smoothing** for pre-recorded video with full trajectory optimisation
- **GPU acceleration** via CUDA OpenCV for optical flow at higher resolutions
- **Adaptive ROI** using semantic segmentation to identify surgical instruments and exclude them from feature tracking
- **H.264 streaming** via GStreamer or ffmpeg for network deployment
- **Multi-camera support** for robotic surgery systems with multiple endoscope feeds
- **Instrument-aware stabilization** that distinguishes between camera motion and instrument motion using separate tracking models
