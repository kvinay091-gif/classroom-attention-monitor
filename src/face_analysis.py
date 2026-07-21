"""
Per-student face analysis using MediaPipe Face Mesh:
  - Eye Aspect Ratio (EAR) -> eyes-closed / sleeping detection
  - solvePnP head pose -> yaw/pitch -> "looking away" detection

Designed to run on a *cropped* person ROI (from the YOLO box), not the
full frame, so MediaPipe only ever sees one face per call.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from . import config

mp_face_mesh = mp.solutions.face_mesh

# Generic 3D face model points (mm), used for solvePnP. Order must match
# config.POSE_LANDMARK_IDX insertion order.
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),         # nose tip
    (0.0, -330.0, -65.0),    # chin
    (-225.0, 170.0, -135.0), # left eye corner (subject's left -> image right)
    (225.0, 170.0, -135.0),  # right eye corner
    (-150.0, -150.0, -125.0),# left mouth corner
    (150.0, -150.0, -125.0), # right mouth corner
], dtype=np.float64)


@dataclass
class FaceReading:
    face_found: bool
    ear: Optional[float] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    eyes_closed: bool = False
    looking_away: bool = False


def _eye_aspect_ratio(landmarks, idx, w, h) -> float:
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx])
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _head_pose(landmarks, w, h):
    idx = config.POSE_LANDMARK_IDX
    image_points = np.array([
        (landmarks[idx["nose_tip"]].x * w, landmarks[idx["nose_tip"]].y * h),
        (landmarks[idx["chin"]].x * w, landmarks[idx["chin"]].y * h),
        (landmarks[idx["left_eye_corner"]].x * w, landmarks[idx["left_eye_corner"]].y * h),
        (landmarks[idx["right_eye_corner"]].x * w, landmarks[idx["right_eye_corner"]].y * h),
        (landmarks[idx["left_mouth_corner"]].x * w, landmarks[idx["left_mouth_corner"]].y * h),
        (landmarks[idx["right_mouth_corner"]].x * w, landmarks[idx["right_mouth_corner"]].y * h),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    return float(yaw), float(pitch)


class FaceAnalyzer:
    def __init__(self):
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.FACE_MESH_MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.FACE_MESH_MIN_DET_CONF,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACK_CONF,
        )

    def analyze(self, person_roi_bgr: np.ndarray) -> FaceReading:
        if person_roi_bgr.size == 0:
            return FaceReading(face_found=False)

        h, w = person_roi_bgr.shape[:2]
        rgb = cv2.cvtColor(person_roi_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return FaceReading(face_found=False)

        landmarks = result.multi_face_landmarks[0].landmark

        left_ear = _eye_aspect_ratio(landmarks, config.LEFT_EYE_IDX, w, h)
        right_ear = _eye_aspect_ratio(landmarks, config.RIGHT_EYE_IDX, w, h)
        ear = (left_ear + right_ear) / 2.0

        yaw, pitch = _head_pose(landmarks, w, h)

        eyes_closed = ear < config.EAR_SLEEP_THRESHOLD
        looking_away = (
            yaw is not None and (
                abs(yaw) > config.YAW_LOOKING_AWAY_DEG
                or abs(pitch) > config.PITCH_LOOKING_AWAY_DEG
            )
        )

        return FaceReading(
            face_found=True,
            ear=ear,
            yaw_deg=yaw,
            pitch_deg=pitch,
            eyes_closed=eyes_closed,
            looking_away=looking_away,
        )

    def close(self):
        self._mesh.close()
