from __future__ import annotations

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


def get_index_fingertip(hand_landmarks, frame_width: int, frame_height: int) -> Optional[tuple[int, int]]:
    if hand_landmarks is None:
        return None

    tip = hand_landmarks.landmark[INDEX_TIP]
    x = int(tip.x * frame_width)
    y = int(tip.y * frame_height)
    return x, y
