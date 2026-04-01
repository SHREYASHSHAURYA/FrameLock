import numpy as np
import cv2
import os


class AdvancedMetrics:
    """Enhanced metrics computation and visualization."""

    def __init__(self):
        self.raw_motion = []
        self.stabilized_motion = []
        self.monocular_motion = []  # For heatmap tracking
        self.frame_qualities = []  # Track which frames are good

    def add_motion(self, raw_dx, raw_dy, stab_dx, stab_dy):
        """Add motion data point."""
        self.raw_motion.append((raw_dx, raw_dy))
        self.stabilized_motion.append((stab_dx, stab_dy))

    def add_frame_quality(self, feature_count, confidence=1.0):
        """Track frame quality based on feature tracking."""
        self.frame_qualities.append(
            {"features": feature_count, "confidence": confidence}
        )

    def compute_detailed_stats(self):
        """Compute comprehensive statistics."""
        if len(self.raw_motion) == 0:
            return None

        raw = np.array(self.raw_motion)
        stab = np.array(self.stabilized_motion)

        raw_magnitude = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
        stab_magnitude = np.sqrt(stab[:, 0] ** 2 + stab[:, 1] ** 2)

        stats = {
            # Magnitude statistics
            "raw_mean": np.mean(raw_magnitude),
            "raw_std": np.std(raw_magnitude),
            "raw_max": np.max(raw_magnitude),
            "stab_mean": np.mean(stab_magnitude),
            "stab_std": np.std(stab_magnitude),
            "stab_max": np.max(stab_magnitude),
            # Component statistics
            "raw_dx_mean": np.mean(np.abs(raw[:, 0])),
            "raw_dy_mean": np.mean(np.abs(raw[:, 1])),
            "stab_dx_mean": np.mean(np.abs(stab[:, 0])),
            "stab_dy_mean": np.mean(np.abs(stab[:, 1])),
            # Improvement
            "improvement_percent": (
                (
                    (np.mean(raw_magnitude) - np.mean(stab_magnitude))
                    / np.mean(raw_magnitude)
                    * 100
                )
                if np.mean(raw_magnitude) > 0
                else 0
            ),
            "frames_processed": len(self.raw_motion),
        }

        return stats

    def generate_motion_heatmap(self, video_width, video_height, grid_size=4):
        """
        Generate motion intensity heatmap by spatial region.

        Returns:
            heatmap (ndarray): Motion intensity grid
        """
        heatmap = np.zeros((grid_size, grid_size))

        if len(self.raw_motion) == 0:
            return heatmap

        raw = np.array(self.raw_motion)
        raw_magnitude = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)

        # Normalize motion to heatmap grid
        for i, intensity in enumerate(raw_magnitude):
            grid_x = min(grid_size - 1, int((i / len(raw_magnitude)) * grid_size))
            heatmap[0, grid_x] = max(heatmap[0, grid_x], intensity)

        # Normalize heatmap to 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

        return heatmap

    def create_heatmap_visualization(self, width=800, height=100):
        """Create visual heatmap image."""
        heatmap = self.generate_motion_heatmap(width, height, grid_size=int(width / 50))

        # Expand to image size
        heatmap_expanded = np.repeat(heatmap[0], height // len(heatmap[0])).reshape(
            height // len(heatmap[0]), -1
        )
        heatmap_expanded = cv2.resize(heatmap_expanded, (width, height))

        # Apply color map
        heatmap_color = cv2.applyColorMap(heatmap_expanded, cv2.COLORMAP_JET)

        # Add labels with better visibility
        cv2.putText(
            heatmap_color,
            "Motion Intensity Timeline",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            heatmap_color,
            "Start",
            (15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            heatmap_color,
            "End",
            (width - 80, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return heatmap_color


class EnhancedPlotter:
    """Enhanced displacement and motion plotting."""

    @staticmethod
    def plot_multi_panel(
        raw_motion, stab_motion, title="Motion Analysis", output_path=None
    ):
        """
        Create multi-panel plot with time-series, phase plot, and histogram.

        Args:
            raw_motion: List of (dx, dy) tuples
            stab_motion: List of (dx, dy) tuples
            title: Plot title
            output_path: Optional path to save image
        """
        if len(raw_motion) == 0:
            # Return empty image if no data
            return np.ones((600, 1200, 3), dtype=np.uint8) * 255

        raw = np.array(raw_motion)
        stab = np.array(stab_motion)

        raw_mag = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
        stab_mag = np.sqrt(stab[:, 0] ** 2 + stab[:, 1] ** 2)

        # Create figure-like structure with 4 subplots
        height = 600
        width = 1200
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Layout: 2x2 grid
        panel_h = height // 2
        panel_w = width // 2

        # Panel 1: Time series magnitude
        EnhancedPlotter._plot_timeseries(
            img, raw_mag, stab_mag, 0, 0, panel_w, panel_h, "Displacement Magnitude"
        )

        # Panel 2: X component
        EnhancedPlotter._plot_timeseries(
            img,
            np.abs(raw[:, 0]),
            np.abs(stab[:, 0]),
            0,
            panel_w,
            panel_w,
            panel_h,
            "X Component",
        )

        # Panel 3: Y component
        EnhancedPlotter._plot_timeseries(
            img,
            np.abs(raw[:, 1]),
            np.abs(stab[:, 1]),
            panel_h,
            0,
            panel_w,
            panel_h,
            "Y Component",
        )

        # Panel 4: Phase plot (X vs Y)
        EnhancedPlotter._plot_phase(
            img, raw, stab, panel_h, panel_w, panel_w, panel_h, "Phase Plot"
        )

        # Add title
        cv2.putText(img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if output_path:
            cv2.imwrite(output_path, img)

        return img

    @staticmethod
    def _plot_timeseries(
        img, raw_data, stab_data, x_offset, y_offset, width, height, label
    ):
        """Plot time series on a panel."""
        n = len(raw_data)
        if n == 0:
            return

        max_val = max(raw_data.max(), stab_data.max(), 1)
        margin = 30
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        x_start = x_offset + margin
        y_start = y_offset + margin

        # Draw axes
        cv2.line(img, (x_start, y_start), (x_start, y_start + plot_h), (0, 0, 0), 1)
        cv2.line(
            img,
            (x_start, y_start + plot_h),
            (x_start + plot_w, y_start + plot_h),
            (0, 0, 0),
            1,
        )

        # Draw data
        for i in range(1, n):
            x0 = x_start + int((i - 1) / n * plot_w)
            x1 = x_start + int(i / n * plot_w)

            y0_raw = y_start + plot_h - int(raw_data[i - 1] / max_val * plot_h)
            y1_raw = y_start + plot_h - int(raw_data[i] / max_val * plot_h)
            y0_stab = y_start + plot_h - int(stab_data[i - 1] / max_val * plot_h)
            y1_stab = y_start + plot_h - int(stab_data[i] / max_val * plot_h)

            cv2.line(img, (x0, y0_raw), (x1, y1_raw), (0, 0, 255), 2)  # Red for raw
            cv2.line(
                img, (x0, y0_stab), (x1, y1_stab), (0, 180, 0), 2
            )  # Green for stabilized

        # Add label and legend
        cv2.putText(
            img,
            label,
            (x_start + 10, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            img,
            "Raw",
            (x_start + 15, y_start + plot_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            img,
            "Stab",
            (x_start + 100, y_start + plot_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 0),
            2,
        )

    @staticmethod
    def _plot_phase(img, raw, stab, x_offset, y_offset, width, height, label):
        """Plot X vs Y phase space."""
        margin = 30
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        x_start = x_offset + margin
        y_start = y_offset + margin

        max_x = max(np.abs(raw[:, 0]).max(), np.abs(stab[:, 0]).max(), 1)
        max_y = max(np.abs(raw[:, 1]).max(), np.abs(stab[:, 1]).max(), 1)

        # Draw axes
        cv2.line(
            img,
            (x_start + plot_w // 2, y_start),
            (x_start + plot_w // 2, y_start + plot_h),
            (128, 128, 128),
            1,
        )
        cv2.line(
            img,
            (x_start, y_start + plot_h // 2),
            (x_start + plot_w, y_start + plot_h // 2),
            (128, 128, 128),
            1,
        )

        # Plot raw points
        for dx, dy in raw:
            px = x_start + plot_w // 2 + int(dx / max_x * plot_w // 2)
            py = y_start + plot_h // 2 - int(dy / max_y * plot_h // 2)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)  # Red

        # Plot stabilized points
        for dx, dy in stab:
            px = x_start + plot_w // 2 + int(dx / max_x * plot_w // 2)
            py = y_start + plot_h // 2 - int(dy / max_y * plot_h // 2)
            cv2.circle(img, (px, py), 2, (0, 180, 0), -1)  # Green

        cv2.putText(
            img,
            label,
            (x_start + 10, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            img,
            "X",
            (x_start + plot_w - 25, y_start + plot_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            img,
            "Y",
            (x_start + plot_w // 2, y_start + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
        )


class TimelinePreview:
    """Generate mini thumbnail timeline of video frames."""

    @staticmethod
    def create_timeline(
        frame_list, video_name="", motion_intensities=None, width=1200, height=100
    ):
        """
        Create thumbnail strip timeline.

        Args:
            frame_list: List of frames or frame paths
            video_name: Video name for label
            motion_intensities: Optional list of motion values for color coding
            width: Output width
            height: Output height
        """
        if len(frame_list) == 0:
            return None

        # Calculate thumbnail dimensions
        thumb_w = max(50, width // len(frame_list))
        thumb_h = height

        timeline = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Add title
        cv2.putText(
            timeline,
            f"Timeline: {video_name}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )

        # Add thumbnails
        for i, frame in enumerate(frame_list):
            x = i * thumb_w

            if isinstance(frame, str):
                if os.path.exists(frame):
                    thumb = cv2.imread(frame)
                    thumb = cv2.resize(thumb, (thumb_w, thumb_h))
            else:
                thumb = cv2.resize(frame, (thumb_w, thumb_h))

            if thumb is not None:
                timeline[:, x : x + thumb_w] = thumb

                # Add motion intensity color bar if available
                if motion_intensities is not None and i < len(motion_intensities):
                    intensity = motion_intensities[i]
                    color = (
                        0,
                        int(255 * (1 - intensity)),
                        int(255 * intensity),
                    )  # Green to Red
                    cv2.rectangle(timeline, (x, 0), (x + thumb_w, 5), color, -1)

        return timeline
