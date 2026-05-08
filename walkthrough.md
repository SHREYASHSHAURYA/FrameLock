# FrameLock — Step-by-Step Walkthrough

> A complete operational guide to running, using, and understanding every component of the FrameLock surgical video stabilization system.

---

## Prerequisites

| Requirement        | Version                 |
| ------------------ | ----------------------- |
| Python             | 3.8 or higher           |
| OpenCV             | `opencv-python` via pip |
| NumPy              | latest stable           |
| Flask + flask-cors | latest stable           |
| Node.js            | 18 or higher            |
| npm                | bundled with Node.js    |

---

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/SHREYASHSHAURYA/FrameLock
cd FrameLock

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install Python dependencies
pip install opencv-python numpy flask flask-cors waitress
```

---

## 2. Prepare Input Videos

Place your `.mp4`, `.avi`, or `.mov` video files into:

```
FrameLock/data/input/
```

These can be any handheld or surgical camera recordings. The system will automatically discover all supported files in this folder. Processed outputs are written to:

```
FrameLock/data/output/
```

---

## 3. Running in Terminal Mode

Terminal mode processes every video in `data/input/` sequentially, opens a live side-by-side OpenCV window, and saves stabilized results to `data/output/`.

```bash
python src/main.py
```

**What you will see:**

- A window opens with the original video on the left and the stabilized version on the right.
- A HUD overlay displays the current frame count, FPS, displacement scores, and tracked feature count.
- A progress bar shows position within the current video.
- After each video finishes, displacement and ROI analysis plots appear sequentially. Press `q` to advance through them.

**Keyboard controls during playback:**

| Key | Action                                              |
| --- | --------------------------------------------------- |
| `0` | Stabilized output (default)                         |
| `1` | Translation demo — shifts frame 60px horizontally   |
| `2` | Rotation demo — rotates frame 0.3 rad around centre |
| `3` | Scaling demo — scales frame 1.3× around centre      |
| `4` | Affine demo — combined rotation + translation       |
| `5` | Perspective demo — quadrilateral warp               |
| `6` | Reflection demo — horizontal flip                   |
| `q` | Skip current video or close current plot            |

Console output after each video:

```
===== surgical1.mp4 =====
Raw motion:        10.40
Stabilized motion: 11.38
Improvement:       -0.98
ROI Raw motion:    12.41
ROI Stabilized:     8.20
ROI Improvement:    4.21
```

Final averaged results across all videos are printed after the last file completes.

---

## 4. Running the Web Dashboard

The web dashboard provides a live browser interface with MJPEG video streams, real-time metric graphs, and full pipeline control.

### Step 1 — Start the API server

```bash
cd src
python api.py
```

The Flask server starts on `http://localhost:5000`. You should see:

```
 * Running on http://0.0.0.0:5000
```

Keep this terminal open. The server must remain running throughout your dashboard session.

### Step 2 — Start the React frontend

Open a second terminal:

```bash
cd ui
npm install        # only needed on first run
npm run dev
```

Vite will print:

```
  VITE ready in Xms
  ➜  Local:   http://localhost:5173/
```

### Step 3 — Open the dashboard

Navigate to `http://localhost:5173` in your browser.

---

## 5. Dashboard Walkthrough

### 5.1 Dashboard Home

The landing page shows two cards — **Dataset Videos** and **Live Camera** — and a status indicator in the sidebar confirming whether the API is reachable. If the indicator shows `API OFFLINE`, verify that `api.py` is running.

Any previously processed videos will appear in a **Recent Results** table at the bottom of the home page.

---

### 5.2 Dataset Videos Page

Click **Dataset Videos** in the sidebar or on the home card.

**Video cards** appear for every file found in `data/input/`. Each card shows:

- Filename, frame count, FPS, duration, and file size
- A status badge: `READY`, `RUNNING`, or `DONE`

**To process a video:**
Click any `READY` card. Only one video can process at a time. While processing, the card shows:

- Live frame counter and progress bar
- A mini displacement sparkline (raw vs stabilised)

**Analysis tabs** appear below the video cards once processing starts or a result is available:

| Tab           | Contents                                                                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **LIVE VIEW** | Side-by-side MJPEG feeds (original left, stabilised right) with ROI rectangle visible on the stabilised feed, full HUD metrics grid, progress bar, and transform mode switcher |
| **GRAPHS**    | Six live charts: displacement magnitude, ROI displacement, X component, Y component, FPS timeline, and motion intensity heatmap                                                |
| **STATS**     | Detailed per-video statistics: mean, std, max for raw and stabilised, dX/dY means, overall improvement percentage, and frame count                                             |
| **BATCH**     | Summary table of all videos processed in the current session                                                                                                                   |

**Zoom:** Each video feed has a `⊞ zoom` button. Clicking it opens the feed as a fullscreen overlay covering the entire page. Press `Esc` or click anywhere outside the image to close.

**Transform mode:** Switch between Stabilized, Translation, Rotation, Scaling, Affine, Perspective, and Reflection at any time using the mode buttons. The change takes effect on the next frame without interrupting processing.

**Stop:** Click **■ STOP** at any time to halt the current video. Metrics up to that point are preserved.

---

### 5.3 Live Camera Page

Click **Live Camera** in the sidebar.

Click **● START CAMERA** to open your default webcam (device index 0). The pipeline runs in real time — no file is required.

The page has three tabs:

| Tab           | Contents                                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------------------- |
| **LIVE VIEW** | Side-by-side webcam feeds with HUD metrics and transform mode switcher                                            |
| **GRAPHS**    | Live sparklines for displacement, ROI, X/Y components, FPS, and motion heatmap — updating continuously            |
| **STATS**     | Full HUD metrics grid: dX, dY, features, raw/stabilised displacement, FPS, ROI raw, ROI stab, and ROI improvement |

Click **■ STOP** to close the webcam and end the session. No output file is written for camera mode.

> Ensure no other application (e.g. video conferencing software) is using the webcam before starting.

---

## 6. API Endpoints Reference

All endpoints are served from `http://localhost:5000`.

| Method | Endpoint                 | Description                                                             |
| ------ | ------------------------ | ----------------------------------------------------------------------- |
| `GET`  | `/status`                | Returns `running`, `source`, and `mode`                                 |
| `GET`  | `/videos`                | Lists all videos in `data/input/` with metadata                         |
| `POST` | `/start`                 | Starts pipeline. Body: `{ "source": "filename.mp4", "mode": "final" }`  |
| `POST` | `/stop`                  | Signals the pipeline to stop after the current frame                    |
| `POST` | `/mode`                  | Changes transform mode live. Body: `{ "mode": "rotation" }`             |
| `GET`  | `/stream`                | SSE stream of `metrics`, `start`, `summary`, `done`, and `error` events |
| `GET`  | `/video_feed/raw`        | MJPEG stream — original frames                                          |
| `GET`  | `/video_feed/stabilized` | MJPEG stream — stabilised frames with ROI rectangle                     |

---

## 7. Understanding the Metrics

| Metric          | Meaning                                                                               |
| --------------- | ------------------------------------------------------------------------------------- |
| **dX / dY**     | Per-frame translation detected between consecutive frames (pixels)                    |
| **Raw Disp.**   | Running mean Euclidean displacement of the raw (unstabilised) motion                  |
| **Stab Disp.**  | Running mean Euclidean displacement after stabilisation                               |
| **Features**    | Number of Shi-Tomasi corner points currently being tracked                            |
| **FPS**         | Frames processed per second (pipeline throughput, not video FPS)                      |
| **ROI Raw**     | Displacement computed only within the anatomy-aware region of interest                |
| **ROI Stab**    | Stabilised displacement within the ROI                                                |
| **ROI Improv.** | `(ROI Raw − ROI Stab) / ROI Raw × 100` — percentage reduction in surgical-site motion |

A positive ROI improvement means the system is successfully reducing shake at the surgical site. The ROI improvement consistently exceeds global improvement because the system focuses correction on the sharpest, most textured region of the frame.

---

## 8. Output Files

After processing a dataset video, a side-by-side `.mp4` is saved to:

```
data/output/<videoname>_stabilized.mp4
```

The output video contains the original frame on the left and the stabilised output on the right, at the original resolution and frame rate.

---

## 9. Common Issues

| Symptom                             | Likely cause                                    | Fix                                                                  |
| ----------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------- |
| `API OFFLINE` badge in dashboard    | `api.py` not running                            | Run `python src/api.py` in a separate terminal                       |
| Video cards not appearing           | No files in `data/input/`                       | Add `.mp4` / `.avi` / `.mov` files and click ↻ Refresh               |
| Stream shows `⚠ stream unavailable` | Browser blocked mixed content or API crashed    | Check terminal for Python errors; reload the page                    |
| Camera fails to start               | Webcam in use by another application            | Close video conferencing apps; ensure device index 0 is available    |
| Processing is slow                  | Large resolution video or many tracked features | Resize input video or reduce feature count in `feature_detection.py` |
| ROI improvement is negative         | Intentional camera movement (panning, zooming)  | Expected — the system corrects shake, not deliberate motion          |
