"""
Draws per-student bounding boxes/status labels on the frame, plus a
summary panel with attendance and the overall class attention score.
"""
import cv2
import numpy as np

STATUS_COLORS = {
    "Sleeping": (0, 0, 255),        # red
    "On Phone": (0, 140, 255),      # orange
    "Distracted": (0, 255, 255),    # yellow
    "Hand Raised": (255, 200, 0),   # cyan-ish
    "Attentive": (0, 200, 0),       # green
}


def draw_student_box(frame, bbox, track_id: int, status: str, score: float):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = STATUS_COLORS.get(status, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {track_id}: {status} ({score:.0f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_summary_panel(frame, class_score: float, attendance: dict, panel_width: int = 260):
    h, w = frame.shape[:2]
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "Classroom Attention", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    score_color = (0, 200, 0) if class_score >= 70 else (0, 165, 255) if class_score >= 45 else (0, 0, 255)
    cv2.putText(panel, f"Class Score: {class_score:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 2, cv2.LINE_AA)

    cv2.putText(panel, f"Present: {attendance['present']}/{attendance['total_tracked']}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.line(panel, (10, 110), (panel_width - 10, 110), (90, 90, 90), 1)

    for label, color in STATUS_COLORS.items():
        y = 130 + list(STATUS_COLORS).index(label) * 22
        cv2.rectangle(panel, (10, y - 12), (26, y + 2), color, -1)
        cv2.putText(panel, label, (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)

    return np.hstack([frame, panel])
