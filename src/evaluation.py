import numpy as np
import cv2


class MotionEvaluator:
    def __init__(self):
        self.raw_motion = []
        self.stabilized_motion = []

    def add_raw(self, dx, dy):
        self.raw_motion.append((dx, dy))

    def add_stabilized(self, dx, dy):
        self.stabilized_motion.append((dx, dy))

    def compute_score(self):
        raw = np.array(self.raw_motion)
        stab = np.array(self.stabilized_motion)

        raw_disp = np.mean(np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2))
        stab_disp = np.mean(np.sqrt(stab[:, 0] ** 2 + stab[:, 1] ** 2))

        return raw_disp, stab_disp

    def plot_displacement(self, title="Displacement"):
        raw = np.array(self.raw_motion)
        stab = np.array(self.stabilized_motion)

        raw_disp = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
        stab_disp = np.sqrt(stab[:, 0] ** 2 + stab[:, 1] ** 2)

        n = len(raw_disp)
        plot_h, plot_w = 300, max(n + 100, 800)
        img = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255

        max_val = max(raw_disp.max(), stab_disp.max(), 1)

        for i in range(1, n):
            x0, x1 = i - 1 + 50, i + 50
            y0_r = int(plot_h - 30 - (raw_disp[i - 1] / max_val) * (plot_h - 60))
            y1_r = int(plot_h - 30 - (raw_disp[i] / max_val) * (plot_h - 60))
            y0_s = int(plot_h - 30 - (stab_disp[i - 1] / max_val) * (plot_h - 60))
            y1_s = int(plot_h - 30 - (stab_disp[i] / max_val) * (plot_h - 60))
            cv2.line(img, (x0, y0_r), (x1, y1_r), (0, 0, 255), 1)
            cv2.line(img, (x0, y0_s), (x1, y1_s), (0, 180, 0), 1)

        cv2.putText(
            img,
            "Raw (red)  Stabilized (green)",
            (50, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.putText(img, title, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
