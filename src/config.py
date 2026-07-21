"""
Central configuration for the classroom attention monitoring pipeline.
Tune thresholds here rather than hunting through module code.
"""

# ---------------------------------------------------------------------------
# YOLOv8 detection
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH = "yolov8n.pt"          # small/fast; swap for yolov8s/m for more accuracy
YOLO_CONF_THRESHOLD = 0.4
YOLO_PERSON_CLASS_ID = 0                 # COCO class id for "person"
YOLO_PHONE_CLASS_ID = 67                 # COCO class id for "cell phone"
TRACKER_CFG = "bytetrack.yaml"           # ships with ultralytics

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh (eyes / head pose)
# ---------------------------------------------------------------------------
FACE_MESH_MAX_FACES = 1                  # per cropped person ROI, we only need 1
FACE_MESH_MIN_DET_CONF = 0.5
FACE_MESH_MIN_TRACK_CONF = 0.5

# Eye landmark indices (MediaPipe 468-point face mesh)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Landmarks used for solvePnP head-pose estimation
POSE_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth_corner": 291,
    "right_mouth_corner": 61,
}

EAR_SLEEP_THRESHOLD = 0.21               # below this = eyes closed
EAR_CONSEC_FRAMES_SLEEP = 20             # ~0.7-1.5s depending on FPS, sustained closure = "sleeping"

YAW_LOOKING_AWAY_DEG = 35                # absolute yaw beyond this = looking away
PITCH_LOOKING_AWAY_DEG = 25              # (e.g. looking down at desk vs. down at phone is handled separately)
LOOK_AWAY_CONSEC_FRAMES = 15

# ---------------------------------------------------------------------------
# MediaPipe Pose (hand raising)
# ---------------------------------------------------------------------------
POSE_MIN_DET_CONF = 0.5
POSE_MIN_TRACK_CONF = 0.5
HAND_RAISE_MARGIN_PX = 15                # wrist must be this many px above shoulder line

# ---------------------------------------------------------------------------
# Phone-usage association
# ---------------------------------------------------------------------------
PHONE_PERSON_IOU_MIN = 0.02              # min overlap between phone box and (expanded) person box
PHONE_CONSEC_FRAMES = 5                  # frames of sustained phone presence to flag "on phone"

# ---------------------------------------------------------------------------
# Attention scoring weights (per student, 0-100 scale)
# ---------------------------------------------------------------------------
SCORE_START = 100.0
PENALTY_SLEEPING = 60.0
PENALTY_PHONE = 45.0
PENALTY_LOOKING_AWAY = 25.0
BONUS_HAND_RAISED = 5.0                  # engagement bonus, capped at SCORE_START
SCORE_SMOOTHING_ALPHA = 0.15             # EMA smoothing factor for displayed score

# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------
ATTENDANCE_MIN_FRAMES_PRESENT = 10       # track must be seen this many frames to count as "attended"

# ---------------------------------------------------------------------------
# Optional CNN eye-state classifier (alternative/backup to EAR heuristic)
# ---------------------------------------------------------------------------
CNN_EYE_MODEL_PATH = "models/eye_state_cnn.pt"
CNN_EYE_IMG_SIZE = 24                    # grayscale 24x24, classic MRL-eye-dataset style input
USE_CNN_EYE_STATE = False                # flip to True once you've trained/placed a model

# ---------------------------------------------------------------------------
# Display / IO
# ---------------------------------------------------------------------------
DRAW_LANDMARKS = False                   # turn on for debugging, off for clean production overlay
WINDOW_NAME = "Classroom Attention Monitor"
