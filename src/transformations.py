import numpy as np
import cv2


def get_translation(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)


def get_rotation(da):
    cos = np.cos(da)
    sin = np.sin(da)
    return np.array([[cos, -sin, 0], [sin, cos, 0]], dtype=np.float32)


def get_scaling(scale):
    return np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)


def apply_affine(frame, transform):
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, transform, (w, h))


def apply_perspective(frame):
    return frame
