import cv2


class VideoReader:
    def __init__(self, input_path):
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise Exception("Error opening video file")

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()


class VideoDisplay:
    def __init__(self, window_name="FrameLock"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, frame):
        h, w = frame.shape[:2]
        scale = min(1280 / w, 720 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        cv2.resizeWindow(self.window_name, new_w, new_h)
        cv2.imshow(self.window_name, resized)

    def wait_key(self, delay=1):
        return cv2.waitKey(delay) & 255

    def destroy(self):
        cv2.destroyAllWindows()
