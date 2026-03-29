import cv2
import numpy as np


def estimate_motion(prev_pts, curr_pts):
    m, _ = cv2.estimateAffine2D(prev_pts, curr_pts)

    if m is None:
        return 0, 0, 0

    dx = m[0, 2]
    dy = m[1, 2]

    da = np.arctan2(m[1, 0], m[0, 0])

    return dx, dy, da
