"""
Fuses per-frame signals (eyes closed, looking away, phone use, hand raised)
into a stable per-student attention score, plus session-wide attendance.

State is kept per YOLO track_id in a rolling deque so a single noisy frame
doesn't flip a student's status; a *sustained* run of frames is required.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from . import config


@dataclass
class StudentState:
    track_id: int
    frames_seen: int = 0
    eyes_closed_run: int = 0
    looking_away_run: int = 0
    phone_run: int = 0
    hand_raised: bool = False
    is_sleeping: bool = False
    is_on_phone: bool = False
    is_looking_away: bool = False
    score_ema: float = config.SCORE_START
    last_bbox: Optional[list] = None

    def update(self, eyes_closed: bool, looking_away: bool, phone_detected: bool, hand_raised: bool):
        self.frames_seen += 1

        self.eyes_closed_run = self.eyes_closed_run + 1 if eyes_closed else 0
        self.looking_away_run = self.looking_away_run + 1 if looking_away else 0
        self.phone_run = self.phone_run + 1 if phone_detected else 0

        self.is_sleeping = self.eyes_closed_run >= config.EAR_CONSEC_FRAMES_SLEEP
        self.is_looking_away = self.looking_away_run >= config.LOOK_AWAY_CONSEC_FRAMES
        self.is_on_phone = self.phone_run >= config.PHONE_CONSEC_FRAMES
        self.hand_raised = hand_raised

        instant_score = config.SCORE_START
        if self.is_sleeping:
            instant_score -= config.PENALTY_SLEEPING
        if self.is_on_phone:
            instant_score -= config.PENALTY_PHONE
        if self.is_looking_away:
            instant_score -= config.PENALTY_LOOKING_AWAY
        if self.hand_raised:
            instant_score += config.BONUS_HAND_RAISED
        instant_score = max(0.0, min(100.0, instant_score))

        alpha = config.SCORE_SMOOTHING_ALPHA
        self.score_ema = alpha * instant_score + (1 - alpha) * self.score_ema

    def status_label(self) -> str:
        if self.is_sleeping:
            return "Sleeping"
        if self.is_on_phone:
            return "On Phone"
        if self.is_looking_away:
            return "Distracted"
        if self.hand_raised:
            return "Hand Raised"
        return "Attentive"


class AttentionEngine:
    def __init__(self):
        self.students: Dict[int, StudentState] = {}

    def get_or_create(self, track_id: int) -> StudentState:
        if track_id not in self.students:
            self.students[track_id] = StudentState(track_id=track_id)
        return self.students[track_id]

    def update_student(self, track_id: int, bbox, eyes_closed: bool,
                        looking_away: bool, phone_detected: bool, hand_raised: bool):
        state = self.get_or_create(track_id)
        state.last_bbox = bbox
        state.update(eyes_closed, looking_away, phone_detected, hand_raised)
        return state

    def class_attention_score(self) -> float:
        active = [s for s in self.students.values() if s.frames_seen > 0]
        if not active:
            return 0.0
        return sum(s.score_ema for s in active) / len(active)

    def attendance_report(self) -> Dict[str, int]:
        """Returns counts of present/total tracked students meeting the
        minimum-frames threshold to be counted as 'attended' this session."""
        present = [s for s in self.students.values()
                   if s.frames_seen >= config.ATTENDANCE_MIN_FRAMES_PRESENT]
        return {
            "present": len(present),
            "total_tracked": len(self.students),
        }

    def summary_rows(self):
        """Row data for a live dashboard table, sorted by track_id."""
        rows = []
        for tid, s in sorted(self.students.items()):
            rows.append({
                "id": tid,
                "status": s.status_label(),
                "score": round(s.score_ema, 1),
                "frames_seen": s.frames_seen,
            })
        return rows
