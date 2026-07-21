"""
AI Classroom Attention Monitoring -- main entry point.

Usage:
    python main.py --source 0                 # webcam
    python main.py --source classroom.mp4      # video file
    python main.py --source rtsp://...         # IP camera
    python main.py --source classroom.mp4 --output annotated.mp4 --no-display

Pipeline per frame:
  1. YOLOv8 detects + tracks persons and phones.
  2. Each person ROI is cropped and passed to MediaPipe Face Mesh (sleep /
     looking-away) and MediaPipe Pose (hand raised).
  3. Phone boxes are IoU-matched against (expanded) person boxes.
  4. AttentionEngine fuses these signals into a smoothed per-student score
     and a class-wide attention score, plus attendance.
  5. Dashboard overlay is drawn and shown / written to disk.
"""
import argparse
import time

import cv2

from src import config
from src.person_tracker import PersonPhoneTracker, iou, expand_box
from src.face_analysis import FaceAnalyzer
from src.pose_analysis import PoseAnalyzer
from src.attention_engine import AttentionEngine
from src.dashboard import draw_student_box, draw_summary_panel


def match_phone_to_persons(persons, phones, frame_w, frame_h):
    """Returns a set of person track_ids that currently have a phone."""
    ids_with_phone = set()
    for person in persons:
        expanded = expand_box(person.bbox, frame_w, frame_h)
        for phone in phones:
            if iou(expanded, phone.bbox) >= config.PHONE_PERSON_IOU_MIN:
                ids_with_phone.add(person.track_id)
                break
    return ids_with_phone


def run(source, output_path=None, display=True, max_frames=None):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    tracker = PersonPhoneTracker()
    face_analyzer = FaceAnalyzer()
    pose_analyzer = PoseAnalyzer()
    engine = AttentionEngine()

    writer = None
    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            h, w = frame.shape[:2]
            persons, phones = tracker.infer(frame)
            phone_ids = match_phone_to_persons(persons, phones, w, h)

            for person in persons:
                if person.track_id is None:
                    continue  # can't maintain rolling state without a stable id

                x1, y1, x2, y2 = [max(0, int(v)) for v in person.bbox]
                roi = frame[y1:y2, x1:x2]

                face_reading = face_analyzer.analyze(roi)
                pose_reading = pose_analyzer.analyze(roi)
                phone_detected = person.track_id in phone_ids

                state = engine.update_student(
                    track_id=person.track_id,
                    bbox=person.bbox,
                    eyes_closed=face_reading.eyes_closed if face_reading.face_found else False,
                    looking_away=face_reading.looking_away if face_reading.face_found else False,
                    phone_detected=phone_detected,
                    hand_raised=pose_reading.hand_raised if pose_reading.pose_found else False,
                )

                draw_student_box(frame, person.bbox, person.track_id,
                                  state.status_label(), state.score_ema)

            class_score = engine.class_attention_score()
            attendance = engine.attendance_report()
            display_frame = draw_summary_panel(frame, class_score, attendance)

            if output_path:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(output_path, fourcc, fps,
                                              (display_frame.shape[1], display_frame.shape[0]))
                writer.write(display_frame)

            if display:
                cv2.imshow(config.WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        face_analyzer.close()
        pose_analyzer.close()

    elapsed = time.time() - t0
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/max(elapsed,1e-6):.1f} FPS)")
    print(f"Final class attention score: {engine.class_attention_score():.1f}")
    print(f"Attendance: {engine.attendance_report()}")
    print("\nPer-student summary:")
    for row in engine.summary_rows():
        print(f"  Student {row['id']:>3}: {row['status']:<12} score={row['score']:<5} frames={row['frames_seen']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Classroom Attention Monitoring")
    parser.add_argument("--source", default="0", help="Webcam index, video file path, or stream URL")
    parser.add_argument("--output", default=None, help="Optional path to save annotated video (.mp4)")
    parser.add_argument("--no-display", action="store_true", help="Don't open a live preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (debugging)")
    args = parser.parse_args()

    run(args.source, output_path=args.output, display=not args.no_display, max_frames=args.max_frames)
