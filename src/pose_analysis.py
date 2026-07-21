"""
Per-student hand-raise detection using MediaPipe Pose, run on the same
cropped person ROI used for face analysis.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

from . import config

mp_pose = mp.solutions.pose


@dataclass
class PoseReading:
    pose_found: bool
    hand_raised: bool = False


class PoseAnalyzer:
    def __init__(self):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.POSE_MIN_DET_CONF,
            min_tracking_confidence=config.POSE_MIN_TRACK_CONF,
        )

    def analyze(self, person_roi_bgr: np.ndarray) -> PoseReading:
        if person_roi_bgr.size == 0:
            return PoseReading(pose_found=False)

        h, w = person_roi_bgr.shape[:2]
        rgb = cv2.cvtColor(person_roi_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return PoseReading(pose_found=False)

        lm = result.pose_landmarks.landmark
        L = mp_pose.PoseLandmark

        def y_px(landmark):
            return landmark.y * h

        left_wrist_y = y_px(lm[L.LEFT_WRIST])
        right_wrist_y = y_px(lm[L.RIGHT_WRIST])
        left_shoulder_y = y_px(lm[L.LEFT_SHOULDER])
        right_shoulder_y = y_px(lm[L.RIGHT_SHOULDER])

        margin = config.HAND_RAISE_MARGIN_PX
        # Smaller y = higher up in image. A raised hand's wrist sits well
        # above (numerically less than) the shoulder line.
        left_raised = left_wrist_y < (left_shoulder_y - margin)
        right_raised = right_wrist_y < (right_shoulder_y - margin)

        return PoseReading(pose_found=True, hand_raised=bool(left_raised or right_raised))

    def close(self):
        self._pose.close()
