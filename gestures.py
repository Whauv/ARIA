from __future__ import annotations

import math
from typing import Optional

from config import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
)


def _is_finger_up(hand_landmarks, tip_idx: int, pip_idx: int) -> bool:
    tip = hand_landmarks.landmark[tip_idx]
    pip = hand_landmarks.landmark[pip_idx]
    return tip.y < pip.y


def is_index_only_up(hand_landmarks) -> bool:
    return (
        _is_finger_up(hand_landmarks, INDEX_TIP, INDEX_PIP)
        and not _is_finger_up(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not _is_finger_up(hand_landmarks, RING_TIP, RING_PIP)
        and not _is_finger_up(hand_landmarks, PINKY_TIP, PINKY_PIP)
    )


def is_index_and_middle_up(hand_landmarks) -> bool:
    return (
        _is_finger_up(hand_landmarks, INDEX_TIP, INDEX_PIP)
        and _is_finger_up(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not _is_finger_up(hand_landmarks, RING_TIP, RING_PIP)
        and not _is_finger_up(hand_landmarks, PINKY_TIP, PINKY_PIP)
    )


def is_closed_fist(hand_landmarks) -> bool:
    return (
        not _is_finger_up(hand_landmarks, INDEX_TIP, INDEX_PIP)
        and not _is_finger_up(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not _is_finger_up(hand_landmarks, RING_TIP, RING_PIP)
        and not _is_finger_up(hand_landmarks, PINKY_TIP, PINKY_PIP)
    )


def get_landmark_point(hand_landmarks, landmark_idx: int, frame_width: int, frame_height: int) -> tuple[int, int]:
    landmark = hand_landmarks.landmark[landmark_idx]
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return x, y


def get_index_fingertip(hand_landmarks, frame_width: int, frame_height: int) -> Optional[tuple[int, int]]:
    if hand_landmarks is None:
        return None

    return get_landmark_point(hand_landmarks, INDEX_TIP, frame_width, frame_height)


def get_fingertip_distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def is_pinching(hand_landmarks, frame_width: int, frame_height: int, threshold: float) -> bool:
    thumb_tip = get_landmark_point(hand_landmarks, THUMB_TIP, frame_width, frame_height)
    index_tip = get_landmark_point(hand_landmarks, INDEX_TIP, frame_width, frame_height)
    return get_fingertip_distance(thumb_tip, index_tip) < threshold


def is_two_hand_resize(hand_landmarks_list) -> bool:
    return hand_landmarks_list is not None and len(hand_landmarks_list) == 2


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def dwell_progress(start_time: float | None, dwell_seconds: float, now: float) -> float:
    if start_time is None:
        return 0.0
    return max(0.0, min(1.0, (now - start_time) / dwell_seconds))


def is_double_pinch(previous_pinch_time: float | None, now: float, threshold_seconds: float) -> bool:
    return previous_pinch_time is not None and now - previous_pinch_time <= threshold_seconds
