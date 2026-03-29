import cv2
import numpy as np

np.random.seed(42)


def convert_to_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def detect_features(gray_frame):
    features = cv2.goodFeaturesToTrack(
        gray_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
    )
    return features
