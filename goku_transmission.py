"""
Instant Transmission webcam: put two fingers on your forehead and you vanish
from your video call. Do it again and you reappear.

How it works
1. Captures an "empty room" photo (you step out of frame for a few seconds).
2. Segments you from the background every frame (MediaPipe).
3. Detects the gesture: index+middle finger up, ring+pinky folded, fingertips on forehead.
4. On gesture: fades you out by blending the empty-room photo over your silhouette, with a quick flash.
5. Sends the result to a virtual camera -> pick "OBS Virtual Camera" in Teams/Zoom/Meet.

Setup (Python 3.12)
    pip install -r requirements.txt    # pins mediapipe 0.10.21: 1.x removed mp.solutions
    Install OBS Studio once (it provides the virtual camera driver), then close OBS.

Run
    python goku_transmission.py            # default webcam
    python goku_transmission.py --cam 1    # another webcam

Keys in the preview window:  b = re-capture background   q = quit
"""
import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

FADE_SECONDS = 0.45      # how long the vanish takes
FLASH_SECONDS = 0.25     # white flash at the moment of teleport
HOLD_SECONDS = 0.35      # how long the pose must be held
THRESH = 0.45            # fingertip-to-forehead distance, as a fraction of face height


def dist(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def finger_up(hand, tip, pip):
    wrist = hand[0]
    return dist(hand[tip], wrist) > dist(hand[pip], wrist) * 1.1


def finger_down(hand, tip, pip):
    wrist = hand[0]
    return dist(hand[tip], wrist) < dist(hand[pip], wrist) * 1.05


def detect_gesture(hand_res, face_res):
    if not hand_res.multi_hand_landmarks or not face_res.multi_face_landmarks:
        return False
    face = [(p.x, p.y) for p in face_res.multi_face_landmarks[0].landmark]
    forehead = ((face[10][0] + face[9][0]) / 2, (face[10][1] + face[9][1]) / 2)
    face_h = dist(face[10], face[152])
    for h in hand_res.multi_hand_landmarks:
        hand = [(p.x, p.y) for p in h.landmark]
        pose = (
            finger_up(hand, 8, 6) and finger_up(hand, 12, 10)
            and finger_down(hand, 16, 14) and finger_down(hand, 20, 18)
        )
        near = min(dist(hand[8], forehead), dist(hand[12], forehead)) < THRESH * face_h
        if pose and near:
            return True
    return False


def capture_background(cap, seconds=5):
    """Countdown, then average a few frames of the empty room."""
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        left = int(seconds - (time.time() - start)) + 1
        view = frame.copy()
        cv2.putText(view, f"Step out of frame... {left}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow("Instant Transmission (preview)", view)
        cv2.waitKey(1)
    frames = []
    for _ in range(15):
        ok, frame = cap.read()
        if ok:
            frames.append(frame.astype(np.float32))
    return np.mean(frames, axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
    face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6)

    background = capture_background(cap)

    visible_target = 1.0   # 1 = visible, 0 = gone
    alpha = 1.0
    flash = 0.0
    held_since = None
    locked = False         # prevents re-triggering until the pose is released
    last = time.time()

    with pyvirtualcam.Camera(width=w, height=h, fps=args.fps,
                             fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        print(f"Virtual camera running: {cam.device}. Select it in your meeting app.")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time()
            dt, last = now - last, now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            face_res = face.process(rgb)

            # --- gesture with hold + toggle ---
            if detect_gesture(hand_res, face_res):
                held_since = held_since or now
                if not locked and now - held_since >= HOLD_SECONDS:
                    visible_target = 0.0 if visible_target == 1.0 else 1.0
                    flash = 1.0
                    locked = True
            else:
                held_since = None
                locked = False

            # --- animate fade + flash ---
            step = dt / FADE_SECONDS
            if alpha < visible_target:
                alpha = min(visible_target, alpha + step)
            elif alpha > visible_target:
                alpha = max(visible_target, alpha - step)
            flash = max(0.0, flash - dt / FLASH_SECONDS)

            # --- person mask (slightly grown + feathered so no halo is left) ---
            if alpha < 1.0:
                m = seg.process(rgb).segmentation_mask
                m = (m > 0.5).astype(np.uint8)
                m = cv2.dilate(m, np.ones((9, 9), np.uint8), iterations=2)
                m = cv2.GaussianBlur(m.astype(np.float32), (21, 21), 0)
                m = (m * (1.0 - alpha))[..., None]          # 0 = keep you, 1 = fully replaced
                out = frame * (1 - m) + background * m
            else:
                out = frame.astype(np.float32)

            if flash > 0:
                out = out + (255 - out) * (flash * 0.8)
            out = np.clip(out, 0, 255).astype(np.uint8)

            cam.send(out)
            cam.sleep_until_next_frame()

            cv2.imshow("Instant Transmission (preview)", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                background = capture_background(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()