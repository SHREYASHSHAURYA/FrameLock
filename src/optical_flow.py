import cv2
import numpy as np


def track_features(prev_gray, curr_gray, prev_points):
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    curr_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )

    good_prev = prev_points[status == 1]
    good_curr = curr_points[status == 1]

    return good_prev, good_curr
