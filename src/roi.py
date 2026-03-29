import cv2
import numpy as np

_prev_cx = None
_prev_cy = None
_frame_count = 0
SMOOTH = 0.95
UPDATE_EVERY = 15


def get_roi(frame):
    global _prev_cx, _prev_cy, _frame_count

    h, w = frame.shape[:2]

    if _prev_cx is None or _frame_count % UPDATE_EVERY == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        block_h = h // 4
        block_w = w // 4

        best_var = -1
        best_r, best_c = 1, 1

        for r in range(1, 3):
            for c in range(1, 3):
                y1 = r * block_h
                y2 = y1 + block_h
                x1 = c * block_w
                x2 = x1 + block_w
                block = gray[y1:y2, x1:x2]
                var = cv2.Laplacian(block, cv2.CV_64F).var()
                if var > best_var:
                    best_var = var
                    best_r, best_c = r, c

        target_cx = int((best_c + 0.5) * block_w)
        target_cy = int((best_r + 0.5) * block_h)

        if _prev_cx is None:
            _prev_cx = target_cx
            _prev_cy = target_cy
        else:
            _prev_cx = int(SMOOTH * _prev_cx + (1 - SMOOTH) * target_cx)
            _prev_cy = int(SMOOTH * _prev_cy + (1 - SMOOTH) * target_cy)

    _frame_count += 1

    roi_w = w // 2
    roi_h = h // 2

    x1 = max(0, _prev_cx - roi_w // 2)
    y1 = max(0, _prev_cy - roi_h // 2)
    x2 = min(w, x1 + roi_w)
    y2 = min(h, y1 + roi_h)

    return x1, y1, x2, y2
