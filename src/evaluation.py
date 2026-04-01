import numpy as np
import cv2
from advanced_metrics import EnhancedPlotter, AdvancedMetrics


class MotionEvaluator:
    def __init__(self):
        self.raw_motion = []
        self.stabilized_motion = []
        self.advanced_metrics = AdvancedMetrics()
        self.frame_details = []  # Track frame numbers for heatmaps

    def add_raw(self, dx, dy):
        self.raw_motion.append((dx, dy))

    def add_stabilized(self, dx, dy):
        self.stabilized_motion.append((dx, dy))

    def compute_score(self):
        if len(self.raw_motion) == 0:
            return 0.0, 0.0

        try:
            raw = np.array(self.raw_motion)
            stab = np.array(self.stabilized_motion)

            raw_disp = np.mean(np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2))
            stab_disp = np.mean(np.sqrt(stab[:, 0] ** 2 + stab[:, 1] ** 2))
        except (IndexError, ValueError):
            # Return safe defaults if data is malformed
            raw_disp = 0.0
            stab_disp = 0.0

        return raw_disp, stab_disp

    def plot_displacement(self, title="Displacement"):
        if len(self.raw_motion) == 0:
            print(f"No motion data for {title}")
            return None

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
            (50, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        cv2.putText(img, title, (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img

    def plot_multi_panel(self, title="Motion Analysis", save_path=None):
        """Create enhanced multi-panel plot with time-series, X/Y components, and phase plot."""
        img = EnhancedPlotter.plot_multi_panel(
            self.raw_motion, self.stabilized_motion, title=title, output_path=save_path
        )
        return img

    def compute_detailed_stats(self):
        """Compute detailed motion statistics including X/Y components and improvement metrics."""
        if len(self.raw_motion) == 0:
            return None

        raw = np.array(self.raw_motion)
        stab = np.array(self.stabilized_motion)

        raw_magnitude = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
        stab_magnitude = np.sqrt(stab[:, 0] ** 2 + stab[:, 1] ** 2)

        stats = {
            "raw_mean": np.mean(raw_magnitude),
            "raw_std": np.std(raw_magnitude),
            "raw_max": np.max(raw_magnitude),
            "stab_mean": np.mean(stab_magnitude),
            "stab_std": np.std(stab_magnitude),
            "stab_max": np.max(stab_magnitude),
            "raw_dx_mean": np.mean(np.abs(raw[:, 0])),
            "raw_dy_mean": np.mean(np.abs(raw[:, 1])),
            "stab_dx_mean": np.mean(np.abs(stab[:, 0])),
            "stab_dy_mean": np.mean(np.abs(stab[:, 1])),
            "improvement_percent": (
                (
                    (np.mean(raw_magnitude) - np.mean(stab_magnitude))
                    / np.mean(raw_magnitude)
                    * 100
                )
                if np.mean(raw_magnitude) > 0
                else 0
            ),
            "frames": len(self.raw_motion),
        }
        return stats

    def create_heatmap_visualization(self, width=800, height=100):
        """Create motion intensity heatmap visualization."""
        if len(self.raw_motion) == 0:
            return None

        raw = np.array(self.raw_motion)
        raw_disp = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)

        n = len(raw_disp)
        heatmap_width = max(n + 100, width)
        img = np.ones((height, heatmap_width, 3), dtype=np.uint8) * 255

        max_val = max(raw_disp.max(), 1)

        for i in range(n):
            intensity = raw_disp[i] / max_val
            x = int(i / n * (heatmap_width - 100)) + 50

            # Color based on intensity (green to red)
            color = (0, int(255 * (1 - intensity)), int(255 * intensity))
            cv2.rectangle(img, (x, 20), (x + 5, height - 20), color, -1)

        cv2.putText(
            img,
            "Motion Intensity Timeline",
            (50, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.line(
            img,
            (50, height // 2),
            (heatmap_width - 50, height // 2),
            (100, 100, 100),
            1,
        )

        return img
