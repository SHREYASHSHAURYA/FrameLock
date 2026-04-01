import cv2
import numpy as np


class HUDOverlay:
    """Display live metrics HUD on video frames."""

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def add_hud(self, frame, metrics):
        """
        Add HUD overlay with metrics.

        Args:
            frame: Input frame
            metrics: Dict with keys: 'frame_count', 'total_frames', 'current_dx', 'current_dy',
                    'raw_displacement', 'stabilized_displacement', 'improvement_pct', 'fps', 'tracked_features'
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Recalculate font scale based on actual frame dimensions
        font_scale = max(0.4, w / 1920)
        thickness = max(1, int(1.5 * font_scale))

        # Text color
        text_color = (0, 255, 0)

        motion_text = f"dX:{metrics.get('current_dx', 0):.1f}  dY:{metrics.get('current_dy', 0):.1f}"
        cv2.putText(
            frame,
            motion_text,
            (30, 40),
            self.font,
            font_scale,
            text_color,
            thickness + 1,
        )

        # Frame counter: next to ORIGINAL label (increased spacing)
        frame_text = (
            f"F: {metrics.get('frame_count', 0)}/{metrics.get('total_frames', 0)}"
        )
        cv2.putText(
            frame,
            frame_text,
            (280, 100),
            self.font,
            font_scale,
            text_color,
            thickness + 1,
        )

        # Raw and stabilized displacement: on top of keyboard controls (increased spacing)
        raw_val = metrics.get("raw_displacement", 0)
        stab_val = metrics.get("stabilized_displacement", 0)
        disp_text = f"Raw: {raw_val:.2f}px  |  Stab: {stab_val:.2f}px"
        cv2.putText(
            frame,
            disp_text,
            (30, 150),
            self.font,
            font_scale,
            text_color,
            thickness + 1,
        )

        # ROI raw and stabilized displacement
        roi_raw_val = metrics.get("roi_raw_displacement", 0)
        roi_stab_val = metrics.get("roi_stabilized_displacement", 0)
        roi_disp_text = (
            f"ROI Raw: {roi_raw_val:.2f}px  |  ROI Stab: {roi_stab_val:.2f}px"
        )
        cv2.putText(
            frame,
            roi_disp_text,
            (30, h - 120),
            self.font,
            font_scale,
            text_color,
            thickness + 1,
        )

        # Improvement %: on top of mode final (increased spacing)
        impl_pct = metrics.get("improvement_pct", 0)
        impl_color = (0, 255, 0) if impl_pct > 0 else (0, 100, 255)
        impl_text = f"ROI Improvement: {impl_pct:.0f}%"
        text_size = cv2.getTextSize(impl_text, self.font, font_scale, thickness)[0]
        cv2.putText(
            frame,
            impl_text,
            (w - text_size[0] - 30, h - 60),
            self.font,
            font_scale,
            impl_color,
            thickness + 1,
        )

        # FPS and tracked features: below keyboard inputs (increased spacing)
        fps_text = f"FPS: {metrics.get('fps', 0):.0f}  |  Features: {metrics.get('tracked_features', 0)}"
        cv2.putText(
            frame,
            fps_text,
            (30, h - 25),
            self.font,
            font_scale,
            text_color,
            thickness + 1,
        )

        return frame

    def add_progress_bar(self, frame, frame_count, total_frames):
        """Add progress bar to frame."""
        h, w = frame.shape[:2]
        bar_height = 10
        bar_y = h - bar_height - 10

        progress = frame_count / max(total_frames, 1)
        bar_width = int((w - 40) * progress)

        cv2.rectangle(
            frame, (20, bar_y), (w - 20, bar_y + bar_height), (100, 100, 100), 1
        )
        cv2.rectangle(
            frame, (20, bar_y), (20 + bar_width, bar_y + bar_height), (0, 255, 0), -1
        )

        return frame


class FeatureTrackingVisualizer:
    """Visualize feature tracking and optical flow."""

    @staticmethod
    def draw_features(frame, features, color=(0, 255, 0), radius=3):
        """Draw detected feature points on frame."""
        if features is None or len(features) == 0:
            return frame

        output = frame.copy()
        for feature in features:
            x, y = feature.ravel()
            cv2.circle(output, (int(x), int(y)), radius, color, -1)
        return output

    @staticmethod
    def draw_optical_flow(
        frame, prev_points, curr_points, color=(0, 255, 255), thickness=2
    ):
        """Draw optical flow vectors between point pairs."""
        if prev_points is None or curr_points is None:
            return frame

        output = frame.copy()
        for prev, curr in zip(prev_points, curr_points):
            x1, y1 = prev.ravel()
            x2, y2 = curr.ravel()
            cv2.arrowedLine(
                output, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness
            )
        return output

    @staticmethod
    def draw_roi_enhanced(frame, x1, y1, x2, y2, show_center=True, show_stats=None):
        """Draw enhanced ROI with center crosshair and optional statistics."""
        output = frame.copy()

        # Draw ROI rectangle with gradient effect
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        # Draw center crosshair
        if show_center:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            size = 20
            cv2.line(
                output,
                (int(cx - size), int(cy)),
                (int(cx + size), int(cy)),
                (0, 255, 0),
                2,
            )
            cv2.line(
                output,
                (int(cx), int(cy - size)),
                (int(cx), int(cy + size)),
                (0, 255, 0),
                2,
            )
            cv2.circle(output, (int(cx), int(cy)), 3, (0, 255, 0), -1)

        # Add statistics if provided
        if show_stats:
            y_text = int(y1) - 30
            for i, (key, value) in enumerate(show_stats.items()):
                text = f"{key}: {value:.2f}"
                cv2.putText(
                    output,
                    text,
                    (int(x1), y_text - i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        return output


class ComparisonPanel:
    """Create side-by-side or overlay comparison panels."""

    @staticmethod
    def create_side_by_side(
        frame1, frame2, label1="ORIGINAL", label2="STABILIZED", label_color=(0, 255, 0)
    ):
        """Create side-by-side comparison with labels."""
        h, w = frame1.shape[:2]
        combined = np.hstack((frame1, frame2))

        font_scale = w / 640
        thickness = max(1, int(2 * font_scale))

        cv2.putText(
            combined,
            label1,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            label_color,
            thickness,
        )
        cv2.putText(
            combined,
            label2,
            (w + 50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            label_color,
            thickness,
        )

        return combined

    @staticmethod
    def create_overlay_diff(frame1, frame2, alpha=0.5):
        """Create overlay difference visualization."""
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Compute absolute difference
        diff = cv2.absdiff(frame1, frame2)

        # Create heatmap of differences
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

        # Overlay difference on original
        output = cv2.addWeighted(frame1, 1 - alpha, diff_heatmap, alpha, 0)
        return output


class KeyboardHint:
    """Display keyboard controls hint on frame."""

    @staticmethod
    def add_controls_hint(frame, mode="final"):
        """Add keyboard controls hint overlay - compact version at bottom."""
        h, w = frame.shape[:2]
        font_scale = max(0.35, w / 2400)
        thickness = max(2, int(1.5 * font_scale))

        hint_h = 80
        hint_y = h - hint_h

        # First-line compact hints with better spacing
        hint_text = (
            "[0]Stab  [1]Trans  [2]Rot  [3]Scale  [4]Aff  [5]Persp  [6]Refl  [Q]Quit"
        )
        cv2.putText(
            frame,
            hint_text,
            (30, hint_y + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        # Current mode indicator on right side - moved lower to not overlap
        mode_colors = {
            "final": (0, 255, 0),
            "translation": (255, 100, 0),
            "rotation": (255, 100, 0),
            "scaling": (255, 100, 0),
            "affine": (255, 100, 0),
            "perspective": (255, 100, 0),
            "reflection": (255, 100, 0),
        }
        mode_text = f"MODE: {mode.upper()}"
        text_size = cv2.getTextSize(
            mode_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
        cv2.putText(
            frame,
            mode_text,
            (w - text_size[0] - 30, hint_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            mode_colors.get(mode, (0, 255, 0)),
            thickness + 1,
        )

        return frame

    @staticmethod
    def add_mode_selector(frame, current_mode=0):
        """Add visual mode selector UI."""
        h, w = frame.shape[:2]
        modes = [
            "Stabilized",
            "Translation",
            "Rotation",
            "Scaling",
            "Affine",
            "Perspective",
            "Reflection",
        ]

        button_w = 100
        button_h = 40
        start_x = 20
        start_y = h - 70
        spacing = 110

        for i, mode_name in enumerate(modes):
            x = start_x + i * spacing
            color = (0, 200, 0) if i == current_mode else (100, 100, 100)
            thickness = 2 if i == current_mode else 1

            cv2.rectangle(
                frame,
                (x, start_y),
                (x + button_w, start_y + button_h),
                color,
                thickness,
            )
            cv2.putText(
                frame,
                str(i),
                (x + 40, start_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness,
            )

        return frame
