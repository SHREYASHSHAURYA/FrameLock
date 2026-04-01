import cv2
import numpy as np
import os


class BatchDashboard:
    """Dashboard for batch video processing with progress tracking and summary statistics."""

    def __init__(self, total_videos):
        self.total_videos = total_videos
        self.processed_videos = 0
        self.video_results = []  # List of result dicts

    def add_video_result(
        self, video_name, raw_disp, stab_disp, features_avg=0, roi_raw=0, roi_stab=0
    ):
        """Log results for a processed video."""
        improvement = ((raw_disp - stab_disp) / raw_disp * 100) if raw_disp > 0 else 0

        result = {
            "name": video_name,
            "raw_displacement": raw_disp,
            "stab_displacement": stab_disp,
            "improvement_percent": improvement,
            "avg_features": features_avg,
            "roi_raw": roi_raw,
            "roi_stab": roi_stab,
        }
        self.video_results.append(result)
        self.processed_videos += 1

    def get_progress(self):
        """Get current progress percentage."""
        return (self.processed_videos / max(self.total_videos, 1)) * 100

    def create_progress_display(self, width=800, height=100):
        """Create progress bar and counters display."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 240

        # Progress bar
        bar_h = 30
        bar_y = 20
        progress_pct = self.get_progress()
        bar_fill = int((width - 40) * progress_pct / 100)

        cv2.rectangle(img, (20, bar_y), (width - 20, bar_y + bar_h), (100, 100, 100), 3)
        cv2.rectangle(img, (20, bar_y), (20 + bar_fill, bar_y + bar_h), (0, 200, 0), -1)

        # Progress text
        progress_text = (
            f"{self.processed_videos}/{self.total_videos} ({progress_pct:.1f}%)"
        )
        cv2.putText(
            img,
            progress_text,
            (width // 2 - 80, bar_y + bar_h // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
        )

        return img

    def create_summary_panel(self, width=1200, height=800):
        """Create comprehensive summary statistics panel."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Title with background
        cv2.rectangle(img, (10, 10), (width - 10, 80), (220, 220, 220), -1)
        cv2.putText(
            img,
            "BATCH PROCESSING SUMMARY",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            4,
        )

        y_offset = 110
        line_height = 50

        # Overall statistics
        if len(self.video_results) > 0:
            raw_disps = [v["raw_displacement"] for v in self.video_results]
            stab_disps = [v["stab_displacement"] for v in self.video_results]
            improvements = [v["improvement_percent"] for v in self.video_results]

            avg_raw = np.mean(raw_disps)
            avg_stab = np.mean(stab_disps)
            avg_improvement = np.mean(improvements)

            # Header with background
            cv2.rectangle(
                img, (20, y_offset - 15), (400, y_offset + 25), (240, 240, 200), -1
            )
            cv2.putText(
                img,
                "Overall Statistics:",
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (30, 30, 100),
                3,
            )
            y_offset += line_height

            # Stats
            stats_text = [
                f"Videos Processed: {len(self.video_results)} / {self.total_videos}",
                f"Avg Raw Displacement: {avg_raw:.3f} px",
                f"Avg Stabilized Displacement: {avg_stab:.3f} px",
                f"Average Improvement: {avg_improvement:.1f}%",
                (
                    f"Total Improvement: {((avg_raw - avg_stab) / avg_raw * 100):.1f}%"
                    if avg_raw > 0
                    else "Total Improvement: N/A"
                ),
            ]

            for text in stats_text:
                cv2.putText(
                    img,
                    text,
                    (60, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 0),
                    2,
                )
                y_offset += line_height

            # Per-video table
            y_offset += 30
            cv2.rectangle(
                img, (20, y_offset - 15), (400, y_offset + 25), (240, 240, 200), -1
            )
            cv2.putText(
                img,
                "Per-Video Results:",
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (30, 30, 100),
                3,
            )
            y_offset += line_height + 10

            # Table headers with background
            headers = ["Video", "Raw (px)", "Stab (px)", "Improvement"]
            header_x = [60, 350, 520, 720]
            cv2.rectangle(
                img,
                (40, y_offset - 20),
                (width - 40, y_offset + 20),
                (200, 200, 200),
                -1,
            )
            for i, header in enumerate(headers):
                cv2.putText(
                    img,
                    header,
                    (header_x[i], y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
            y_offset += 40

            # Table rows
            for idx, result in enumerate(self.video_results):
                if y_offset > height - 60:
                    break

                # Alternating row background
                if idx % 2 == 0:
                    cv2.rectangle(
                        img,
                        (40, y_offset - 20),
                        (width - 40, y_offset + 20),
                        (240, 240, 240),
                        -1,
                    )

                video_name = os.path.splitext(result["name"])[0][:18]
                row_text = [
                    video_name,
                    f"{result['raw_displacement']:.2f}",
                    f"{result['stab_displacement']:.2f}",
                    f"{result['improvement_percent']:.1f}%",
                ]

                # Color code improvement
                improvement_color = (
                    (0, 200, 0) if result["improvement_percent"] > 0 else (0, 0, 200)
                )

                for i, text in enumerate(row_text):
                    color = improvement_color if i == 3 else (0, 0, 0)
                    cv2.putText(
                        img,
                        text,
                        (header_x[i], y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                y_offset += line_height

        return img


class VideoCard:
    """Individual video processing card with metrics."""

    @staticmethod
    def create_card(
        video_name,
        thumbnail=None,
        status="Processing",
        metrics=None,
        width=300,
        height=200,
    ):
        """
        Create a card displaying video processing information.

        Args:
            video_name: Name of the video
            thumbnail: Optional frame thumbnail
            status: Processing status (Processing, Complete, Error)
            metrics: Dict with metrics (raw_displacement, stab_displacement, improvement_percent)
            width: Card width
            height: Card height
        """
        card = np.ones((height, width, 3), dtype=np.uint8) * 240

        # Border
        color = (
            (0, 200, 0)
            if status == "Complete"
            else (200, 100, 0) if status == "Processing" else (0, 0, 200)
        )
        cv2.rectangle(card, (0, 0), (width - 1, height - 1), color, 4)

        # Thumbnail or placeholder
        if thumbnail is not None:
            thumb_resized = cv2.resize(thumbnail, (width - 4, 100))
            card[2:102, 2 : width - 2] = thumb_resized
            y_offset = 115
        else:
            cv2.rectangle(card, (2, 2), (width - 2, 102), (150, 150, 150), -1)
            cv2.putText(
                card,
                "No Thumb",
                (width // 2 - 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 100),
                2,
            )
            y_offset = 115

        # Video name
        truncated_name = video_name[:18] if len(video_name) > 18 else video_name
        cv2.putText(
            card,
            truncated_name,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        y_offset += 25

        # Status
        cv2.putText(
            card,
            f"Status: {status}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        y_offset += 23

        # Metrics if available
        if metrics:
            cv2.putText(
                card,
                f"Raw: {metrics.get('raw_displacement', 0):.2f}px",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            y_offset += 20
            cv2.putText(
                card,
                f"Stab: {metrics.get('stab_displacement', 0):.2f}px",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                2,
            )
            y_offset += 20
            improvement = metrics.get("improvement_percent", 0)
            improvement_color = (0, 200, 0) if improvement > 0 else (0, 0, 200)
            cv2.putText(
                card,
                f"Improvement: {improvement:.1f}%",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                improvement_color,
                2,
            )

        return card


class StatisticsPanel:
    """Display comprehensive statistics of video processing."""

    @staticmethod
    def create_before_after_comparison(
        raw_motion, stab_motion, title="Motion Before/After", width=1000, height=400
    ):
        """Create detailed before vs after comparison visualization."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        if len(raw_motion) == 0:
            cv2.putText(
                img,
                "No motion data available",
                (width // 2 - 150, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )
            return img

        raw = np.array(raw_motion)
        stab = np.array(stab_motion)

        raw_mag = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
        stab_mag = np.sqrt(stab[:, 0] ** 2 + stab[:, 1] ** 2)

        # Title with background
        cv2.rectangle(img, (10, 10), (width - 10, 50), (220, 220, 220), -1)
        cv2.putText(img, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 0), 3)

        # Split display
        mid_x = width // 2

        # Left side: Raw motion
        margin = 50
        plot_h = height - 140
        plot_w = mid_x - 70

        max_val = max(raw_mag.max(), stab_mag.max(), 1)

        # Draw raw motion
        x_start = margin
        y_start = 80
        cv2.rectangle(
            img, (x_start, y_start), (x_start + plot_w, y_start + plot_h), (0, 0, 0), 2
        )
        cv2.putText(
            img,
            "RAW MOTION",
            (x_start + 10, y_start + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        for i in range(1, len(raw_mag)):
            x0 = x_start + int((i - 1) / len(raw_mag) * plot_w)
            x1 = x_start + int(i / len(raw_mag) * plot_w)
            y0 = y_start + plot_h - int(raw_mag[i - 1] / max_val * plot_h)
            y1 = y_start + plot_h - int(raw_mag[i] / max_val * plot_h)
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # Draw stabilized motion
        x_start = mid_x + margin
        cv2.rectangle(
            img, (x_start, y_start), (x_start + plot_w, y_start + plot_h), (0, 0, 0), 2
        )
        cv2.putText(
            img,
            "STABILIZED MOTION",
            (x_start + 10, y_start + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 0),
            2,
        )

        for i in range(1, len(stab_mag)):
            x0 = x_start + int((i - 1) / len(stab_mag) * plot_w)
            x1 = x_start + int(i / len(stab_mag) * plot_w)
            y0 = y_start + plot_h - int(stab_mag[i - 1] / max_val * plot_h)
            y1 = y_start + plot_h - int(stab_mag[i] / max_val * plot_h)
            cv2.line(img, (x0, y0), (x1, y1), (0, 200, 0), 2)

        # Statistics at bottom
        raw_mean = np.mean(raw_mag)
        stab_mean = np.mean(stab_mag)
        improvement = ((raw_mean - stab_mean) / raw_mean * 100) if raw_mean > 0 else 0

        stats_y = y_start + plot_h + 55
        cv2.rectangle(
            img,
            (margin, stats_y - 30),
            (width - margin, stats_y + 15),
            (240, 240, 240),
            -1,
        )
        cv2.putText(
            img,
            f"Raw Mean: {raw_mean:.3f} px  |  Stab Mean: {stab_mean:.3f} px  |  Improvement: {improvement:.1f}%",
            (margin + 15, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        return img


class MotionIntensityHeatmap:
    """Generate motion intensity heatmaps for spatial analysis."""

    @staticmethod
    def create_regional_heatmap(
        frame_height, frame_width, motion_by_region, grid_size=4, title="Motion Heatmap"
    ):
        """
        Create spatial heatmap of motion across video regions.

        Args:
            frame_height: Video frame height
            frame_width: Video frame width
            motion_by_region: 2D array of motion intensities [grid_y, grid_x]
            grid_size: Grid resolution
            title: Heatmap title
        """
        heatmap_h = grid_size * 40 + 80
        heatmap_w = grid_size * 40 + 80
        img = np.ones((heatmap_h, heatmap_w, 3), dtype=np.uint8) * 255

        # Title
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cell_size = 40
        start_x = 40
        start_y = 80

        # Draw grid with color intensity
        if motion_by_region is not None and motion_by_region.size > 0:
            max_intensity = motion_by_region.max() if motion_by_region.max() > 0 else 1

            for r in range(grid_size):
                for c in range(grid_size):
                    intensity = (
                        motion_by_region[r, c]
                        if r < motion_by_region.shape[0]
                        and c < motion_by_region.shape[1]
                        else 0
                    )
                    normalized = intensity / max_intensity

                    # Create color: green (good) to red (bad)
                    color = (0, int(255 * (1 - normalized)), int(255 * normalized))

                    x1 = start_x + c * cell_size
                    y1 = start_y + r * cell_size
                    cv2.rectangle(
                        img, (x1, y1), (x1 + cell_size, y1 + cell_size), color, -1
                    )
                    cv2.rectangle(
                        img, (x1, y1), (x1 + cell_size, y1 + cell_size), (0, 0, 0), 2
                    )

                    # Add text label
                    if intensity > 0:
                        text = f"{intensity:.1f}"
                        cv2.putText(
                            img,
                            text,
                            (x1 + 5, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2,
                        )

        # Legend
        legend_y = start_y + grid_size * cell_size + 30
        cv2.putText(
            img,
            "Green (Low Motion) -> Red (High Motion)",
            (start_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
        )

        return img
