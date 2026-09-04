import cv2
import mediapipe as mp
import numpy as np
import math
import time
import platform
from pynput.keyboard import Key, Controller

CAMERA_INDEX       = 1
DEAD_ZONE_DEG      = 12
RELEASE_ZONE_DEG   = 6
SOFT_ZONE_DEG       = 25
FLIP_CAMERA        = True
SHOW_ANGLE         = True
MIN_DETECTION_CONF = 0.7
MIN_TRACKING_CONF  = 0.5
GRACE_FRAMES       = 8
OPEN_FINGER_THRESH = 3

# --- Smoothing / sensitivity (adjustable at runtime with +/-) ---
SMOOTHING_ALPHA    = 0.35   # exponential smoothing factor (0-1, higher = snappier)
SENS_STEP          = 0.05
SENS_MIN           = 0.05
SENS_MAX           = 1.0

# --- Watermark ---
AUTHOR_NAME        = "Sandesh Chaitanya"
WATERMARK_TEXT     = f"Made by {AUTHOR_NAME}"
SHOW_WATERMARK      = True

# --- Camera reconnect ---
CAMERA_RETRY_DELAY  = 1.0
MAX_CAMERA_RETRIES  = 5

CLR_WHEEL   = (80, 200, 255)
CLR_LEFT    = (60, 120, 255)
CLR_RIGHT   = (50, 220, 140)
CLR_NEUTRAL = (200, 200, 200)
CLR_TEXT    = (255, 255, 255)
CLR_ACCENT  = (0, 180, 255)
CLR_HAND_L  = (255, 130, 60)
CLR_HAND_R  = (60, 230, 130)
CLR_ACCEL   = (50, 220, 100)
CLR_BRAKE   = (0, 60, 255)

keyboard   = Controller()
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def is_open_hand(hand_landmarks):
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    extended = sum(
        1 for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    )
    return extended >= OPEN_FINGER_THRESH


class SteeringController:
    def __init__(self):
        self.keys_held      = {Key.left: False, Key.right: False, Key.up: False, Key.down: False}
        self.smoothed_angle = None
        self.alpha          = SMOOTHING_ALPHA
        self.sensitivity    = 1.0   # multiplies dead/soft zone thresholds; lower = more sensitive
        self.paused         = False

    def _press(self, key):
        if self.paused:
            return
        if not self.keys_held[key]:
            keyboard.press(key)
            self.keys_held[key] = True

    def _release(self, key):
        if not self.keys_held[key]:
            return
        keyboard.release(key)
        self.keys_held[key] = False

    def release_all(self):
        for key in list(self.keys_held.keys()):
            try:
                keyboard.release(key)
            except Exception:
                pass
            self.keys_held[key] = False
        self.smoothed_angle = None

    def set_paused(self, paused):
        self.paused = paused
        if paused:
            self.release_all()

    def adjust_sensitivity(self, delta):
        self.sensitivity = float(np.clip(self.sensitivity + delta, SENS_MIN, SENS_MAX))
        return self.sensitivity

    def smooth_angle(self, raw_angle):
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            self.smoothed_angle = self.alpha * raw_angle + (1 - self.alpha) * self.smoothed_angle
        return self.smoothed_angle

    def update_steer(self, left_wrist, right_wrist):
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        raw_angle_rad = math.atan2(dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)
        angle = self.smooth_angle(raw_angle_deg)

        dead_zone = DEAD_ZONE_DEG * self.sensitivity
        soft_zone = max(SOFT_ZONE_DEG * self.sensitivity, dead_zone + 1)
        release_zone = RELEASE_ZONE_DEG * self.sensitivity

        direction = "STRAIGHT"
        if angle < -dead_zone:
            direction = "LEFT"
        elif angle > dead_zone:
            direction = "RIGHT"
        elif self.keys_held[Key.left] and angle > -release_zone:
            direction = "STRAIGHT"
        elif self.keys_held[Key.right] and angle < release_zone:
            direction = "STRAIGHT"

        strength = 0.0
        if direction == "LEFT":
            strength = min(1.0, (abs(angle) - dead_zone) / (soft_zone - dead_zone))
            self._press(Key.left)
            self._release(Key.right)
        elif direction == "RIGHT":
            strength = min(1.0, (abs(angle) - dead_zone) / (soft_zone - dead_zone))
            self._press(Key.right)
            self._release(Key.left)
        else:
            self._release(Key.left)
            self._release(Key.right)

        return angle, direction, strength

    def update_throttle(self, left_open, right_open):
        both_open  = left_open and right_open
        both_fist  = (not left_open) and (not right_open)

        if both_fist:
            self._press(Key.up)
            self._release(Key.down)
            return "ACCEL"
        elif both_open:
            self._press(Key.down)
            self._release(Key.up)
            return "BRAKE"
        else:
            self._release(Key.up)
            self._release(Key.down)
            return "NEUTRAL"


def draw_steering_wheel(frame, center, angle_deg, direction, strength):
    h, w = frame.shape[:2]
    radius = int(min(w, h) * 0.10)
    cx, cy = center

    color = CLR_NEUTRAL
    if direction == "LEFT":
        color = CLR_LEFT
    elif direction == "RIGHT":
        color = CLR_RIGHT

    cv2.circle(frame, (cx + 3, cy + 3), radius, (0, 0, 0), 4)
    cv2.circle(frame, (cx, cy), radius, color, 3)

    for sa in [0, 120, 240]:
        rad = math.radians(sa - angle_deg)
        x1 = int(cx + radius * 0.4 * math.cos(rad))
        y1 = int(cy - radius * 0.4 * math.sin(rad))
        x2 = int(cx + radius * 0.95 * math.cos(rad))
        y2 = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.circle(frame, (cx, cy), 6, color, -1)

    if direction != "STRAIGHT":
        start_a = -30 if direction == "RIGHT" else 150
        end_a   =  30 if direction == "RIGHT" else 210
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_a, end_a, color, 5)


def draw_hud(frame, angle, direction, strength, throttle_mode, both_hands_visible, left_open, right_open, fps, sensitivity=1.0):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 160), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bar_w = int(w * 0.5)
    bar_h = 14
    bar_x = (w - bar_w) // 2
    bar_y = h - 110
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

    mid = bar_x + bar_w // 2
    cv2.rectangle(frame, (mid - 2, bar_y - 4), (mid + 2, bar_y + bar_h + 4), (180, 180, 180), -1)

    fill_len = int((bar_w // 2) * strength)
    if direction == "LEFT" and fill_len > 0:
        cv2.rectangle(frame, (mid - fill_len, bar_y), (mid, bar_y + bar_h), CLR_LEFT, -1)
    elif direction == "RIGHT" and fill_len > 0:
        cv2.rectangle(frame, (mid, bar_y), (mid + fill_len, bar_y + bar_h), CLR_RIGHT, -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    dir_color = CLR_LEFT if direction == "LEFT" else (CLR_RIGHT if direction == "RIGHT" else CLR_NEUTRAL)
    cv2.putText(frame, "<- LEFT",  (bar_x, bar_y - 10),               font, 0.45, CLR_LEFT,  1)
    cv2.putText(frame, "RIGHT ->", (bar_x + bar_w - 80, bar_y - 10),  font, 0.45, CLR_RIGHT, 1)
    cv2.putText(frame, direction,  (mid - 30, bar_y + bar_h + 28),    font, 0.8,  dir_color, 2)

    if SHOW_ANGLE:
        cv2.putText(frame, f"{angle:+.1f} deg", (bar_x, h - 80), font, 0.55, CLR_TEXT, 1)

    throttle_color = CLR_ACCEL if throttle_mode == "ACCEL" else (CLR_BRAKE if throttle_mode == "BRAKE" else CLR_NEUTRAL)
    throttle_label = {
        "ACCEL":   "ACCEL [UP]",
        "BRAKE":   "BRAKE [DOWN]",
        "NEUTRAL": "NEUTRAL",
    }[throttle_mode]

    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), (30, 30, 40), -1)
    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), throttle_color, 2)
    cv2.putText(frame, throttle_label, (bar_x + 10, h - 48), font, 0.65, throttle_color, 2)

    l_label = "OPEN" if left_open else "FIST"
    r_label = "OPEN" if right_open else "FIST"
    l_color = CLR_BRAKE if left_open else CLR_ACCEL
    r_color = CLR_BRAKE if right_open else CLR_ACCEL
    cv2.putText(frame, f"L:{l_label}", (bar_x + bar_w + 10, h - 100), font, 0.5, l_color, 1)
    cv2.putText(frame, f"R:{r_label}", (bar_x + bar_w + 10, h - 80),  font, 0.5, r_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 30), font, 0.55, CLR_ACCENT, 1)
    cv2.putText(frame, f"SENS: {sensitivity:.2f}", (w - 130, 55), font, 0.45, CLR_ACCENT, 1)

    status       = "BOTH HANDS DETECTED" if both_hands_visible else "SHOW BOTH HANDS"
    status_color = (60, 220, 60) if both_hands_visible else (0, 80, 255)
    cv2.putText(frame, status, (10, 30), font, 0.55, status_color, 1)

    draw_steering_wheel(frame, (w - 80, h - 80), angle, direction, strength)
    draw_watermark(frame)


def draw_watermark(frame):
    if not SHOW_WATERMARK:
        return
    h, w = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    (tw, th), _ = cv2.getTextSize(WATERMARK_TEXT, font, font_scale, thickness)
    x = w - tw - 12
    y = h - 12
    cv2.putText(frame, WATERMARK_TEXT, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, WATERMARK_TEXT, (x, y), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)


def draw_help_overlay(frame):
    h, w = frame.shape[:2]
    lines = [
        "Q / ESC  - Quit",
        "P        - Pause / Resume",
        "H        - Toggle this help",
        "+ / -    - Steering sensitivity",
        "FIST both hands = Accelerate",
        "OPEN both hands = Brake",
        "Tilt hands to steer",
    ]
    box_w, box_h = 300, 26 * len(lines) + 20
    ox, oy = 10, 45
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + box_w, oy + box_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (ox, oy), (ox + box_w, oy + box_h), CLR_ACCENT, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (ox + 12, oy + 26 + i * 26), font, 0.48, CLR_TEXT, 1, cv2.LINE_AA)


def draw_paused_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "PAUSED - press P to resume"
    (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
    cv2.putText(frame, text, ((w - tw) // 2, h // 2), font, 0.9, CLR_ACCENT, 2, cv2.LINE_AA)


def draw_hand_connection(frame, lw, rw):
    lx, ly = lw
    rx, ry = rw
    cv2.line(frame, (lx, ly), (rx, ry), (30, 100, 200), 8)
    cv2.line(frame, (lx, ly), (rx, ry), CLR_ACCENT, 2)
    cv2.circle(frame, (lx, ly), 10, CLR_HAND_L, -1)
    cv2.circle(frame, (rx, ry), 10, CLR_HAND_R, -1)
    cv2.circle(frame, (lx, ly), 13, CLR_HAND_L, 2)
    cv2.circle(frame, (rx, ry), 13, CLR_HAND_R, 2)
    mx = (lx + rx) // 2
    my = (ly + ry) // 2
    cv2.circle(frame, (mx, my), 7, CLR_WHEEL, -1)


def main():
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

    def open_camera():
        c = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not c.isOpened():
            c = cv2.VideoCapture(CAMERA_INDEX)
        return c

    cap = open_camera()
    retries = 0
    while not cap.isOpened() and retries < MAX_CAMERA_RETRIES:
        print(f"[WARN] Camera not ready, retrying ({retries + 1}/{MAX_CAMERA_RETRIES})...")
        time.sleep(CAMERA_RETRY_DELAY)
        cap = open_camera()
        retries += 1

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("  -> macOS: System Settings > Privacy & Security > Camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    controller = SteeringController()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )

    conn_style     = mp_drawing.DrawingSpec(color=(80, 80, 100), thickness=1)
    landmark_style = mp_drawing.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=2)

    prev_time     = time.time()
    angle         = 0.0
    direction     = "STRAIGHT"
    strength      = 0.0
    throttle_mode = "NEUTRAL"
    left_open     = False
    right_open    = False
    lost_frames   = 0
    show_help     = False
    consecutive_read_fails = 0

    print("=" * 55)
    print(f"  Virtual Steering Wheel  |  {WATERMARK_TEXT}")
    print("=" * 55)
    print("  FIST  = Accelerate (UP)    OPEN = Brake (DOWN)")
    print("  Tilt hands LEFT/RIGHT to steer — works in any mode")
    print("  Q / ESC = Quit   P = Pause   H = Help   +/- = Sensitivity")
    print("=" * 55)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_fails += 1
                if consecutive_read_fails > 60:
                    print("[WARN] Losing camera feed, attempting to reconnect...")
                    cap.release()
                    cap = open_camera()
                    consecutive_read_fails = 0
                time.sleep(0.01)
                continue
            consecutive_read_fails = 0

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            both_visible = False

            if controller.paused:
                pass
            elif results.multi_hand_landmarks and results.multi_handedness:
                hand_data = {}

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_style, conn_style)

                    wrist  = hand_landmarks.landmark[0]
                    wx     = int(wrist.x * w)
                    wy     = int(wrist.y * h)
                    opened = is_open_hand(hand_landmarks)
                    hand_data[label] = (wrist.x, wrist.y, wx, wy, opened)

                if "Left" in hand_data and "Right" in hand_data:
                    both_visible = True
                    lost_frames  = 0

                    lx_n, ly_n, lx_px, ly_px, left_open  = hand_data["Left"]
                    rx_n, ry_n, rx_px, ry_px, right_open = hand_data["Right"]

                    draw_hand_connection(frame, (lx_px, ly_px), (rx_px, ry_px))
                    angle, direction, strength = controller.update_steer((lx_n, ly_n), (rx_n, ry_n))
                    throttle_mode = controller.update_throttle(left_open, right_open)
                else:
                    lost_frames += 1
                    if lost_frames >= GRACE_FRAMES:
                        controller.release_all()
                        angle, direction, strength = 0.0, "STRAIGHT", 0.0
                        throttle_mode = "NEUTRAL"
                        left_open = right_open = False
            else:
                lost_frames += 1
                if lost_frames >= GRACE_FRAMES:
                    controller.release_all()
                    angle, direction, strength = 0.0, "STRAIGHT", 0.0
                    throttle_mode = "NEUTRAL"
                    left_open = right_open = False

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame, angle, direction, strength, throttle_mode, both_visible,
                     left_open, right_open, fps, controller.sensitivity)

            if show_help:
                draw_help_overlay(frame)
            if controller.paused:
                draw_paused_overlay(frame)

            cv2.imshow("Virtual Steering Wheel", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('p'), ord('P')):
                controller.set_paused(not controller.paused)
            elif key in (ord('h'), ord('H')):
                show_help = not show_help
            elif key in (ord('+'), ord('=')):
                s = controller.adjust_sensitivity(-SENS_STEP)
                print(f"[INFO] Sensitivity: {s:.2f}")
            elif key in (ord('-'), ord('_')):
                s = controller.adjust_sensitivity(SENS_STEP)
                print(f"[INFO] Sensitivity: {s:.2f}")

    finally:
        controller.release_all()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. All keys released.")


if __name__ == "__main__":
    main()
