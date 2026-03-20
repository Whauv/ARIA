# ARIA

ARIA stands for Augmented Reality Intelligence Assistant.

This project is a Python prototype for real-time webcam-based air drawing with OpenCV and MediaPipe Hands.

## Current Features

- Live webcam capture with `cv2.VideoCapture(0)`
- Mirrored webcam feed for natural interaction
- Real-time hand landmark detection with MediaPipe Hands
- Index fingertip tracking with exponential smoothing
- Air drawing onto a separate canvas
- Gesture-based draw and pause controls
- Finish gesture that converts a drawing into a transparent sprite
- Multi-sprite rendering with alpha compositing and `z_index` ordering

## Files

```text
.
├── main.py
├── canvas.py
├── gestures.py
├── config.py
├── sprite.py
└── README.md
```

## Gestures

- Index finger only up: draw
- Index and middle fingers up: pause drawing
- Closed fist held for 1.5 seconds: convert the current drawing into a sprite
- Press `q`: quit

## Install

```bash
pip install opencv-python mediapipe numpy
```

## Run

```bash
python main.py
```
