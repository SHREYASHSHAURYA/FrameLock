import os
import cv2
import numpy as np
import time
from video_io import VideoReader, VideoDisplay
from feature_detection import convert_to_grayscale, detect_features
from optical_flow import track_features
from motion_estimation import estimate_motion
from smoothing import Trajectory
from evaluation import MotionEvaluator
from roi import get_roi
from transformations import (
    get_translation,
    get_rotation,
    get_scaling,
    apply_affine,
    apply_perspective,
)
from visualization_utils import (
    HUDOverlay,
    FeatureTrackingVisualizer,
    ComparisonPanel,
    KeyboardHint,
)
from advanced_metrics import AdvancedMetrics
from batch_dashboard import BatchDashboard, StatisticsPanel


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "..", "data", "input")

    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    all_raw = []
    all_stab = []
    all_roi_raw = []
    all_roi_stab = []

    # Initialize batch dashboard
    batch_dashboard = BatchDashboard(len(video_files))

    for video_name in video_files:
        video_path = os.path.join(input_folder, video_name)

        reader = VideoReader(video_path)
        display = VideoDisplay("FrameLock")

        name = os.path.splitext(video_name)[0]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(
            base_dir, "..", "data", "output", f"{name}_output.mp4"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        ret, temp_frame = reader.read_frame()
        if not ret:
            continue

        h, w = temp_frame.shape[:2]
        fps = reader.cap.get(cv2.CAP_PROP_FPS) or 30

        reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

        trajectory = Trajectory()
        evaluator = MotionEvaluator()
        roi_evaluator = MotionEvaluator()
        import roi as roi_module

        roi_module._prev_cx = None
        roi_module._prev_cy = None
        roi_module._frame_count = 0

        mode = "final"

        # Initialize HUD and visualization tools
        hud_overlay = HUDOverlay(w, h)
        frame_tracking_viz = FeatureTrackingVisualizer()
        start_time = time.time()
        total_frames_estimate = int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300

        ret, prev_frame = reader.read_frame()
        if not ret:
            continue

        prev_gray = convert_to_grayscale(prev_frame)

        x1, y1, x2, y2 = get_roi(prev_frame)
        roi_gray = prev_gray[y1:y2, x1:x2]

        roi_points = detect_features(roi_gray)

        if roi_points is not None:
            roi_points[:, :, 0] += x1
            roi_points[:, :, 1] += y1

        prev_points = roi_points

        frame_count = 0

        while True:
            ret, frame = reader.read_frame()
            if not ret:
                break

            curr_gray = convert_to_grayscale(frame)

            if prev_points is None or len(prev_points) < 50:
                prev_points = detect_features(prev_gray)

            if prev_points is not None:
                prev_pts, curr_pts = track_features(prev_gray, curr_gray, prev_points)
                x1, y1, x2, y2 = get_roi(frame)

                roi_prev = []
                roi_curr = []

                for p, c in zip(prev_pts, curr_pts):
                    px, py = p.ravel()
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        roi_prev.append(p)
                        roi_curr.append(c)

                if len(roi_prev) < 10:
                    prev_points = detect_features(prev_gray)
                    combined_frame = np.hstack((frame, frame))
                    raw_disp, stab_disp = evaluator.compute_score()
                    hud_metrics = {
                        "frame_count": frame_count,
                        "total_frames": total_frames_estimate,
                        "current_dx": 0,
                        "current_dy": 0,
                        "raw_displacement": raw_disp,
                        "stabilized_displacement": stab_disp,
                        "improvement_pct": 0,
                        "fps": frame_count / max(time.time() - start_time, 0.001),
                        "tracked_features": 0,
                    }
                    combined_frame = hud_overlay.add_hud(combined_frame, hud_metrics)
                    combined_frame = KeyboardHint.add_controls_hint(
                        combined_frame, mode
                    )
                    display.show(combined_frame)
                    out.write(combined_frame)
                    prev_gray = curr_gray.copy()
                    continue

                roi_prev = np.array(roi_prev)
                roi_curr = np.array(roi_curr)

                if prev_pts is None or len(prev_pts) < 10:
                    prev_points = detect_features(prev_gray)
                    combined_frame = np.hstack((frame, frame))
                    raw_disp, stab_disp = evaluator.compute_score()
                    hud_metrics = {
                        "frame_count": frame_count,
                        "total_frames": total_frames_estimate,
                        "current_dx": 0,
                        "current_dy": 0,
                        "raw_displacement": raw_disp,
                        "stabilized_displacement": stab_disp,
                        "improvement_pct": 0,
                        "fps": frame_count / max(time.time() - start_time, 0.001),
                        "tracked_features": 0,
                    }
                    combined_frame = hud_overlay.add_hud(combined_frame, hud_metrics)
                    combined_frame = KeyboardHint.add_controls_hint(
                        combined_frame, mode
                    )
                    display.show(combined_frame)
                    out.write(combined_frame)
                    prev_gray = curr_gray.copy()
                    continue

                dx, dy, da = estimate_motion(prev_pts, curr_pts)

                global_dx, global_dy, _ = estimate_motion(prev_pts, curr_pts)
                evaluator.add_raw(global_dx, global_dy)
                roi_dx, roi_dy, _ = estimate_motion(roi_prev, roi_curr)
                roi_evaluator.add_raw(roi_dx, roi_dy)

                x, y, a = trajectory.update(dx, dy, da)

                if True:
                    smoothed = trajectory.smooth_kalman()
                    sx, sy, sa = smoothed[-1]

                    if len(trajectory.trajectory) > 1:
                        prev_sx, prev_sy, prev_sa = smoothed[-2]

                        diff_x = (sx - prev_sx) - dx
                        diff_y = (sy - prev_sy) - dy
                        diff_a = (sa - prev_sa) - da
                    else:
                        diff_x, diff_y, diff_a = 0, 0, 0

                    evaluator.add_stabilized(diff_x, diff_y)
                    roi_evaluator.add_stabilized(diff_x, diff_y)

                    # ===== REAL STABILIZATION =====
                    t_mat = get_translation(diff_x, diff_y)
                    r_mat = get_rotation(diff_a)

                    h, w = frame.shape[:2]

                    roi_cx = (x1 + x2) / 2
                    roi_cy = (y1 + y2) / 2

                    T1 = np.array([[1, 0, -roi_cx], [0, 1, -roi_cy]], dtype=np.float32)
                    T2 = np.array([[1, 0, roi_cx], [0, 1, roi_cy]], dtype=np.float32)

                    M = np.vstack([r_mat, [0, 0, 1]])
                    T = np.vstack([t_mat, [0, 0, 1]])

                    T1_3 = np.array(
                        [[1, 0, -roi_cx], [0, 1, -roi_cy], [0, 0, 1]], dtype=np.float32
                    )

                    T2_3 = np.array(
                        [[1, 0, roi_cx], [0, 1, roi_cy], [0, 0, 1]], dtype=np.float32
                    )

                    combined = r_mat.copy()
                    combined[:, 2] += t_mat[:, 2]

                    fh, fw = frame.shape[:2]
                    stabilized = apply_affine(frame, combined)
                    border = int(min(fh, fw) * 0.05)
                    stabilized = stabilized[border : fh - border, border : fw - border]
                    stabilized = cv2.resize(stabilized, (fw, fh))
                    cv2.rectangle(
                        stabilized,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        4,
                    )

                    # ===== DEMO =====
                    h, w = frame.shape[:2]
                    cx, cy = w / 2, h / 2

                    def center_matrix(M):
                        T1 = np.array(
                            [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32
                        )
                        T2 = np.array(
                            [[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32
                        )
                        M3 = np.vstack([M, [0, 0, 1]])
                        return (T2 @ M3 @ T1)[:2, :]

                    if mode == "translation":
                        output = apply_affine(frame, get_translation(60, 0))
                    elif mode == "rotation":
                        M = center_matrix(get_rotation(0.3))
                        output = apply_affine(frame, M)
                    elif mode == "scaling":
                        M = center_matrix(get_scaling(1.3))
                        output = apply_affine(frame, M)
                    elif mode == "affine":
                        A = get_rotation(0.3)
                        A[:, 2] += [40, 30]
                        M = center_matrix(A)
                        output = apply_affine(frame, M)
                    elif mode == "perspective":
                        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                        dst = np.float32([[60, 40], [w - 60, 0], [0, h - 60], [w, h]])
                        P = cv2.getPerspectiveTransform(src, dst)
                        output = cv2.warpPerspective(frame, P, (w, h))
                    elif mode == "reflection":
                        output = cv2.flip(frame, 1)
                    else:
                        zoom = 1.15
                        zh, zw = int(fh / zoom), int(fw / zoom)
                        zy, zx = (fh - zh) // 2, (fw - zw) // 2
                        output = cv2.resize(
                            stabilized[zy : zy + zh, zx : zx + zw], (fw, fh)
                        )

                    combined_frame = np.hstack((frame, output))

                    # Add labels
                    label_font_scale = max(0.8, fw / 1024)
                    label_thickness = 3

                    # Left label
                    cv2.putText(
                        combined_frame,
                        "ORIGINAL",
                        (15, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        (0, 255, 0),
                        label_thickness,
                    )

                    # Right label
                    cv2.putText(
                        combined_frame,
                        "STABILIZED",
                        (fw + 15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        (0, 255, 0),
                        label_thickness,
                    )

                    # Add HUD overlay with metrics
                    raw_disp, stab_disp = evaluator.compute_score()
                    roi_raw, roi_stab = roi_evaluator.compute_score()
                    improvement_pct = (
                        ((roi_raw - roi_stab) / roi_raw * 100) if roi_raw > 0 else 0
                    )
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / max(elapsed_time, 0.001)

                    hud_metrics = {
                        "frame_count": frame_count,
                        "total_frames": total_frames_estimate,
                        "current_dx": dx,
                        "current_dy": dy,
                        "raw_displacement": raw_disp,
                        "stabilized_displacement": stab_disp,
                        "roi_raw_displacement": roi_raw,
                        "roi_stabilized_displacement": roi_stab,
                        "improvement_pct": improvement_pct,
                        "fps": current_fps,
                        "tracked_features": (
                            len(prev_pts) if prev_pts is not None else 0
                        ),
                    }
                    combined_frame = hud_overlay.add_hud(combined_frame, hud_metrics)
                    combined_frame = hud_overlay.add_progress_bar(
                        combined_frame, frame_count, total_frames_estimate
                    )
                    combined_frame = KeyboardHint.add_controls_hint(
                        combined_frame, mode
                    )

                    display.show(combined_frame)
                    out.write(combined_frame)
                else:
                    combined_frame = np.hstack((frame, frame))
                    raw_disp, stab_disp = evaluator.compute_score()
                    hud_metrics = {
                        "frame_count": frame_count,
                        "total_frames": total_frames_estimate,
                        "current_dx": 0,
                        "current_dy": 0,
                        "raw_displacement": raw_disp,
                        "stabilized_displacement": stab_disp,
                        "improvement_pct": 0,
                        "fps": frame_count / max(time.time() - start_time, 0.001),
                        "tracked_features": 0,
                    }
                    combined_frame = hud_overlay.add_hud(combined_frame, hud_metrics)
                    combined_frame = KeyboardHint.add_controls_hint(
                        combined_frame, mode
                    )
                    display.show(combined_frame)
                    out.write(combined_frame)

                prev_points = curr_pts.reshape(-1, 1, 2)
            else:
                combined_frame = np.hstack((frame, frame))
                raw_disp, stab_disp = evaluator.compute_score()
                hud_metrics = {
                    "frame_count": frame_count,
                    "total_frames": total_frames_estimate,
                    "current_dx": 0,
                    "current_dy": 0,
                    "raw_displacement": raw_disp,
                    "stabilized_displacement": stab_disp,
                    "improvement_pct": 0,
                    "fps": frame_count / max(time.time() - start_time, 0.001),
                    "tracked_features": 0,
                }
                combined_frame = hud_overlay.add_hud(combined_frame, hud_metrics)
                combined_frame = KeyboardHint.add_controls_hint(combined_frame, mode)
                display.show(combined_frame)
                out.write(combined_frame)

            prev_gray = curr_gray.copy()
            frame_count += 1

            key = cv2.waitKey(1)

            if key == ord("1"):
                mode = "translation"
            elif key == ord("2"):
                mode = "rotation"
            elif key == ord("3"):
                mode = "scaling"
            elif key == ord("4"):
                mode = "affine"
            elif key == ord("5"):
                mode = "perspective"
            elif key == ord("6"):
                mode = "reflection"
            elif key == ord("0"):
                mode = "final"
            elif key == ord("q"):
                break

        reader.release()
        cv2.destroyAllWindows()
        if out is not None:
            out.release()

        # Displacement plots - ROI only
        roi_disp = roi_evaluator.plot_displacement(
            title=f"{video_name} - ROI Displacement"
        )
        if roi_disp is not None:
            cv2.imshow(f"{video_name} - Displacement", roi_disp)
            while True:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    cv2.destroyWindow(f"{video_name} - Displacement")
                    break

        # Enhanced visualization features (features 1, 4, 5, 8, 9, 10, 12)

        # Feature 4 & 5: Enhanced multi-panel plot - ROI only
        roi_analysis = roi_evaluator.plot_multi_panel(
            title=f"{video_name} - ROI Analysis"
        )
        if roi_analysis is not None:
            cv2.imshow(f"{video_name} - Analysis", roi_analysis)
            while True:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    cv2.destroyWindow(f"{video_name} - Analysis")
                    break

        # Feature 9: Motion heatmap visualization - ROI only
        roi_heatmap_viz = roi_evaluator.create_heatmap_visualization()
        if roi_heatmap_viz is not None:
            cv2.imshow(f"{video_name} - Motion Heatmap", roi_heatmap_viz)
            while True:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    cv2.destroyWindow(f"{video_name} - Motion Heatmap")
                    break

        raw_disp, stab_disp = evaluator.compute_score()
        roi_raw, roi_stab = roi_evaluator.compute_score()

        # Skip visualizations if no motion data
        if len(evaluator.raw_motion) == 0:
            print(f"No motion data for {video_name}, skipping visualizations")
        else:
            # Feature 12: Before/After comparison panel - ROI only
            roi_comparison_panel = StatisticsPanel.create_before_after_comparison(
                roi_evaluator.raw_motion,
                roi_evaluator.stabilized_motion,
                title=f"{video_name} - ROI Before/After Motion Comparison",
            )
            cv2.imshow(f"{video_name} - Comparison", roi_comparison_panel)
            while True:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    cv2.destroyWindow(f"{video_name} - Comparison")
                    break

        # Feature 8: Summary statistics panel
        detailed_stats = evaluator.compute_detailed_stats()

        print(f"\n===== {video_name} =====")
        print("Raw motion:", raw_disp)
        print("Stabilized motion:", stab_disp)
        print("Improvement:", raw_disp - stab_disp)

        # Print detailed statistics (feature 8)
        if detailed_stats:
            print(f"\nDetailed Statistics:")
            print(
                f"  Raw - Mean: {detailed_stats['raw_mean']:.3f}, Std: {detailed_stats['raw_std']:.3f}, Max: {detailed_stats['raw_max']:.3f}"
            )
            print(
                f"  Stabilized - Mean: {detailed_stats['stab_mean']:.3f}, Std: {detailed_stats['stab_std']:.3f}, Max: {detailed_stats['stab_max']:.3f}"
            )
            print(f"  Improvement: {detailed_stats['improvement_percent']:.1f}%")

        print("ROI Raw motion:", roi_raw)
        print("ROI Stabilized motion:", roi_stab)
        print("ROI Improvement:", roi_raw - roi_stab)

        all_raw.append(raw_disp)
        all_stab.append(stab_disp)
        all_roi_raw.append(roi_raw)
        all_roi_stab.append(roi_stab)

        # Feature 13: Add to batch dashboard
        avg_features = len(prev_pts) if prev_pts is not None else 0
        batch_dashboard.add_video_result(
            video_name, raw_disp, stab_disp, avg_features, roi_raw, roi_stab
        )

    print("\n===== FINAL AVERAGE =====")

    if len(all_raw) > 0:
        print("Avg Raw motion:", np.mean(all_raw))
        print("Avg Stabilized motion:", np.mean(all_stab))
        print("Avg Improvement:", np.mean(all_raw) - np.mean(all_stab))
    else:
        print("No raw motion data collected")

    if len(all_roi_raw) > 0:
        print("\nAvg ROI Raw motion:", np.mean(all_roi_raw))
        print("Avg ROI Stabilized motion:", np.mean(all_roi_stab))
        print("Avg ROI Improvement:", np.mean(all_roi_raw) - np.mean(all_roi_stab))
    else:
        print("\nNo ROI motion data collected")

    # Feature 13: Display batch processing dashboard and summary statistics panel
    # Show progress panel
    progress_panel = batch_dashboard.create_progress_display()
    cv2.imshow("Batch Processing - Complete", progress_panel)
    while True:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            cv2.destroyWindow("Batch Processing - Complete")
            break

    # Show summary panel
    summary_panel = batch_dashboard.create_summary_panel()
    cv2.imshow("Batch Processing Summary", summary_panel)
    while True:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            cv2.destroyWindow("Batch Processing Summary")
            break


if __name__ == "__main__":
    main()
