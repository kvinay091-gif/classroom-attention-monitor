# AI Classroom Attention Monitoring

Real-time computer vision pipeline that detects sleeping, phone usage,
looking away, and hand-raising per student, then rolls those up into
attendance and an overall class attention score.

## Architecture

| Signal | Model | Module |
|---|---|---|
| Person detection + tracking | YOLOv8 (ByteTrack) | `src/person_tracker.py` |
| Phone detection | YOLOv8 (COCO "cell phone" class) | `src/person_tracker.py` |
| Eye closure / sleeping | MediaPipe Face Mesh (Eye Aspect Ratio) | `src/face_analysis.py` |
| Looking away | MediaPipe Face Mesh + solvePnP head pose | `src/face_analysis.py` |
| Hand raising | MediaPipe Pose | `src/pose_analysis.py` |
| Eye state (optional, more robust) | Small CNN (PyTorch) | `src/cnn_eye_state.py` |
| Fusion / scoring / attendance | rolling per-student state machine | `src/attention_engine.py` |
| Overlay / dashboard | OpenCV drawing | `src/dashboard.py` |

**Why this stack:** YOLOv8 gives fast, robust person + object detection with
built-in multi-object tracking (so each student keeps a stable ID across
frames). MediaPipe gives free, real-time facial landmarks and body pose
without training your own detector. The CNN is there as a drop-in upgrade
for eye-state classification if lighting, glasses, or camera angle make the
EAR geometric heuristic unreliable — train it on your own footage once you
have labeled crops.

## Setup

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run via
`ultralytics`. For better accuracy at the cost of speed, swap
`YOLO_MODEL_PATH` in `src/config.py` to `yolov8s.pt` or `yolov8m.pt`.

## Run

```bash
# Webcam
python main.py --source 0

# Video file, save annotated output, no live window (e.g. on a server)
python main.py --source classroom.mp4 --output annotated.mp4 --no-display
```

Press `q` in the preview window to stop early. A per-student summary and
attendance report print to the console when the run ends.

## How each detection works

- **Sleeping**: Eye Aspect Ratio (EAR) computed from 6 eye landmarks per
  eye. EAR drops sharply when eyes close. A *sustained* run of
  `EAR_CONSEC_FRAMES_SLEEP` low-EAR frames (not just one blink) triggers
  the "Sleeping" flag.
- **Phone usage**: YOLOv8 detects phone-shaped objects; each phone box is
  IoU-matched against a slightly expanded student bounding box, so a phone
  held near the chest/lap still counts. Sustained detection over
  `PHONE_CONSEC_FRAMES` avoids single-frame false positives.
- **Looking away**: 6 facial landmarks feed `cv2.solvePnP` against a
  generic 3D face model to get yaw/pitch. Beyond a configurable angle
  threshold, sustained over several frames, flags "Distracted".
- **Hand raising**: MediaPipe Pose wrist vs. shoulder y-coordinate — wrist
  meaningfully above the shoulder line = hand raised.
- **Attendance**: any track ID seen for at least `ATTENDANCE_MIN_FRAMES_PRESENT`
  frames counts as present for the session.
- **Attention score**: each student starts at 100 and takes weighted
  penalties (sleeping > phone > looking away) with a small engagement
  bonus for hand raising, smoothed with an exponential moving average so
  the on-screen number doesn't jitter frame to frame. The class score is
  the average of all currently-tracked students' scores.

All thresholds/weights live in `src/config.py` — tune there rather than in
the module code.

## Tuning tips

- **False "sleeping" flags**: raise `EAR_SLEEP_THRESHOLD` slightly or
  increase `EAR_CONSEC_FRAMES_SLEEP` if students blinking normally get
  flagged.
- **Phone detection misses**: lower `YOLO_CONF_THRESHOLD` a bit, or switch
  to a larger YOLO model — small/angled phones are the hardest case for
  the nano model.
- **Looking-away too sensitive**: classroom seating means many students
  aren't facing the camera even when attentive to the board off to one
  side — recalibrate `YAW_LOOKING_AWAY_DEG` per camera placement, or add a
  per-seat baseline offset.
- **Multiple cameras**: run one `PersonPhoneTracker` + `AttentionEngine`
  pair per camera/section and merge attendance reports; track IDs are not
  shared across separate model instances.

## Optional: training the CNN eye-state classifier

```bash
python -m src.cnn_eye_state train --data-dir /path/to/dataset
```

Expects `dataset/open/*.png` and `dataset/closed/*.png` (24x24 or larger
grayscale/color eye crops — public datasets like MRL Eye Dataset or CEW
work well). After training, set `USE_CNN_EYE_STATE = True` and
`CNN_EYE_MODEL_PATH` in `src/config.py`, then wire
`EyeStateClassifier.predict_closed()` into `face_analysis.py` in place of
(or alongside) the EAR check.

## Important notes on responsible use

This is a technically real-time, per-student surveillance system. Before
deploying it in an actual classroom:

- Check your institution's/region's laws on biometric monitoring, student
  privacy (e.g. FERPA in the US, GDPR in the EU), and consent requirements
  for recording minors.
- Be transparent with students and guardians about what's tracked and how
  scores are used — attention scores are noisy proxies, not ground truth,
  and shouldn't solely drive grading or disciplinary decisions.
- Consider running this as an aggregate/anonymized classroom-level signal
  (just the class score, no per-student IDs shown) if individual tracking
  isn't necessary for your use case.

## Known limitations

- Occlusion (students facing away, heads down on desks) will drop face/pose
  readings — the engine treats "no face found" as neutral, not automatically
  "distracted," to avoid false positives; adjust if you want the opposite
  default.
- YOLO's "cell phone" class doesn't distinguish a phone in hand from one
  sitting on a desk; the IoU association helps but isn't perfect.
- Track IDs can occasionally switch (ID swaps) during heavy occlusion,
  which will reset that student's rolling counters.
