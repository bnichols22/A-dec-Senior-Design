# ==============================================================
# File: centroid_tracking.py
# Purpose:
#   Moving "stable region" tracking for gimbal control.
#   - Keeps a small neutral box centered on last STABLE face position (anchor)
#   - When mouth centroid exits the box, emits ONE incremental delta
#     (Δyaw_deg, Δpitch_deg, Δroll_deg) from anchor -> current
#   - Enforces a MIN_SEND_INTERVAL so the controller is not spammed
#   - Re-anchors only after the face becomes stable again (dwell + low speed)
# ==============================================================

import os, sys, time, warnings, math, statistics
import cv2
import mediapipe as mp
from collections import deque

# -------------------- Config --------------------
LOG_PATH = os.path.expanduser("~/senior_design/camera/testing/face_track_log.txt")

# Camera FOVs (deg) — approximate; set to your lens
FOV_H_DEG = 95.0
FOV_V_DEG = 60.0

# Neutral box size as fraction of HALF-frame (± around anchor)
NEUTRAL_BOX_NORM_W = 0.06   # tighter box = fewer commands (try 0.04–0.08)
NEUTRAL_BOX_NORM_H = 0.06

# Stability detection (must be "calm" before re-anchoring)
VEL_THRESH_DEG_S   = 2.5     # average angular speed below this
POS_STD_THRESH_PX  = 2.5     # centroid stddev (px) below this
STABLE_DWELL_S     = 0.45    # must remain calm for this many seconds
WINDOW_SEC         = 0.6     # size of history window for stats

# Command emission control
MIN_SEND_INTERVAL_S = 2.0    # rate limit: seconds between sends
MIN_STEP_DEG        = 0.5    # ignore tiny deltas
ROLL_MIN_STEP_DEG   = 1.0    # roll is noisier

# Smoothing
SMOOTH_ALPHA = 0.25          # EMA for centroid

# Misc
DRAW = True
CAM_INDEX = 0
MAX_LOST_FRAMES = 10

# ----------------- Quiet noisy logs -------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.stderr = open(LOG_PATH, "w")

# ------------------ Helpers ---------------------
def ema_point(curr, prev, alpha):
    if prev is None: return curr
    return (alpha*curr[0] + (1-alpha)*prev[0],
            alpha*curr[1] + (1-alpha)*prev[1])

def pixels_to_deg(dx_px, dy_px, w, h, fov_h, fov_v):
    half_w, half_h = w/2.0, h/2.0
    return (dx_px/half_w)*(fov_h/2.0), (dy_px/half_h)*(fov_v/2.0)

def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def box_from_anchor(anchor, w, h):
    cx, cy = anchor
    nbx = NEUTRAL_BOX_NORM_W * (w/2.0)
    nby = NEUTRAL_BOX_NORM_H * (h/2.0)
    return (cx - nbx, cy - nby, cx + nbx, cy + nby)

def inside_box(p, box):
    x, y = p
    l, t, r, b = box
    return l <= x <= r and t <= y <= b

# Rolling history with timestamps
class TimedHist:
    def __init__(self, win_sec):
        self.win = win_sec
        self.buf = deque()

    def add(self, t, val):
        self.buf.append((t, val))
        self._trim(t)

    def values(self):
        return [v for (_, v) in self.buf]

    def _trim(self, tnow):
        cut = tnow - self.win
        while self.buf and self.buf[0][0] < cut:
            self.buf.popleft()

# --------------- MediaPipe init -----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------- Camera -----------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("ERROR: Unable to open camera", CAM_INDEX)
    sys.exit(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ----------------- State ------------------------
prev_smoothed     = None
prev_roll_deg     = None
stable_anchor     = None      # moving anchor (neutral box center)
last_sent_time    = 0.0
pending_reanchor  = False     # set True right after sending; re-anchors once stable

lost = 0
t0 = time.time()

# Histories for stability metrics
pos_hist_x = TimedHist(WINDOW_SEC)
pos_hist_y = TimedHist(WINDOW_SEC)
speed_hist = TimedHist(WINDOW_SEC)  # angular speed magnitude (deg/s)

print("# t_s d_yaw d_pitch d_roll | anchor_x anchor_y stable can_send")

while True:
    ok, frame = cap.read()
    if not ok:
        print("WARN: frame grab failed"); break

    now = time.time()
    h, w = frame.shape[:2]

    # --- Detect mouth centroid + roll ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    centroid = None
    roll_now_deg = None

    if res.multi_face_landmarks:
        fl = res.multi_face_landmarks[0]
        mouth_idxs = [13, 14, 61, 291]
        pts = []
        for idx in mouth_idxs:
            x = int(fl.landmark[idx].x * w)
            y = int(fl.landmark[idx].y * h)
            pts.append((x, y))
        if pts:
            avg_x = sum(p[0] for p in pts) / len(pts)
            avg_y = sum(p[1] for p in pts) / len(pts)
            centroid = (avg_x, avg_y)

            p_left  = (int(fl.landmark[61].x * w), int(fl.landmark[61].y * h))
            p_right = (int(fl.landmark[291].x * w), int(fl.landmark[291].y * h))
            roll_now_deg = angle_deg(p_left, p_right)

    # ----- Handle lost detection -----
    if centroid is None:
        lost += 1
        if lost > MAX_LOST_FRAMES:
            prev_smoothed = None
            prev_roll_deg = None
            # keep stable_anchor; you can also clear it if you prefer hard reset
            pos_hist_x.buf.clear()
            pos_hist_y.buf.clear()
            speed_hist.buf.clear()
        t = now - t0
        print(f"{t:.3f} +0.000 +0.000 +0.000 | {0.0:.1f} {0.0:.1f} 0 0")
        if DRAW:
            cv2.imshow("Stable Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
        continue

    lost = 0
    smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

    # --- Update stability histories ---
    pos_hist_x.add(now, smoothed[0])
    pos_hist_y.add(now, smoothed[1])

    # speed: deg/s based on delta to previous smoothed (if available)
    if prev_smoothed is None:
        speed = 0.0
    else:
        dx_px = smoothed[0] - prev_smoothed[0]
        dy_px = smoothed[1] - prev_smoothed[1]
        dvx_deg, dvy_deg = pixels_to_deg(dx_px, dy_px, w, h, FOV_H_DEG, FOV_V_DEG)
        # estimate per-frame speed; dt ~ frame period—good enough for stability gating
        # you can refine with a tracked dt if you want
        speed = math.hypot(dvx_deg, dvy_deg) * 30.0  # assume ~30 fps => deg/s
    speed_hist.add(now, speed)
    prev_smoothed = smoothed

    # --- Compute stability stats ---
    xs = pos_hist_x.values()
    ys = pos_hist_y.values()
    pos_std = 999.0
    if len(xs) >= 6 and len(ys) >= 6:
        # population stddev for robustness
        pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))

    speeds = speed_hist.values()
    vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0
    is_stable_now = (vel_med < VEL_THRESH_DEG_S) and (pos_std < POS_STD_THRESH_PX)

    # --- Initialize anchor to first detection if needed ---
    if stable_anchor is None:
        stable_anchor = smoothed
        stable_enter_time = now  # start dwell timer
    else:
        # If calm, track dwell; else reset dwell timer
        if is_stable_now:
            if 'stable_enter_time' not in locals():
                stable_enter_time = now
        else:
            stable_enter_time = now  # restart dwell timer

    stable_for = now - (stable_enter_time if 'stable_enter_time' in locals() else now)
    can_reanchor = is_stable_now and (stable_for >= STABLE_DWELL_S)

    # Current neutral box around anchor
    box = box_from_anchor(stable_anchor, w, h)
    is_outside = not inside_box(smoothed, box)

    # ---- Rate limit for controller ----
    can_send = (now - last_sent_time) >= MIN_SEND_INTERVAL_S

    # ---- Default deltas (no command) ----
    d_yaw = d_pitch = d_roll = 0.0

    # ---- Emit one incremental delta when leaving the box ----
    if is_outside and can_send:
        dx_px = smoothed[0] - stable_anchor[0]
        dy_px = smoothed[1] - stable_anchor[1]
        d_yaw, d_pitch = pixels_to_deg(dx_px, dy_px, w, h, FOV_H_DEG, FOV_V_DEG)

        if roll_now_deg is not None and prev_roll_deg is not None:
            d_roll = roll_now_deg - prev_roll_deg

        # Apply minimum step thresholds
        if abs(d_yaw)   < MIN_STEP_DEG:      d_yaw = 0.0
        if abs(d_pitch) < MIN_STEP_DEG:      d_pitch = 0.0
        if abs(d_roll)  < ROLL_MIN_STEP_DEG: d_roll = 0.0

        if d_yaw != 0.0 or d_pitch != 0.0 or d_roll != 0.0:
            # ---- YOUR SEND HOOK ----
            # on_delta(d_yaw, d_pitch, d_roll)
            last_sent_time = now
            pending_reanchor = True  # wait for stability before moving anchor

    # ---- Re-anchor after it settles in the new place ----
    if pending_reanchor and can_reanchor:
        stable_anchor = smoothed
        pending_reanchor = False
        # clear histories to avoid “old jitter” bias
        pos_hist_x.buf.clear()
        pos_hist_y.buf.clear()
        speed_hist.buf.clear()
        stable_enter_time = now  # reset dwell baseline

    # Update roll baseline
    if roll_now_deg is not None:
        prev_roll_deg = roll_now_deg

    # ---- Telemetry ----
    t = now - t0
    ax, ay = stable_anchor
    print(f"{t:.3f} {d_yaw:+.3f} {d_pitch:+.3f} {d_roll:+.3f} | {ax:.1f} {ay:.1f} {1 if is_stable_now else 0} {1 if can_send else 0}")

    # ---- UI ----
    if DRAW:
        l, t_, r, b = map(int, box)
        cv2.rectangle(frame, (l, t_), (r, b), (120, 200, 120), 1)
        cv2.drawMarker(frame, (int(ax), int(ay)), (0, 200, 0), cv2.MARKER_CROSS, 12, 2)
        cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

        info1 = f"stable:{int(is_stable_now)} dwell:{stable_for:.2f}s vel_med:{vel_med:.1f} std:{pos_std:.1f}"
        info2 = f"send_gate:{int(can_send)} dYaw:{d_yaw:+.2f} dPitch:{d_pitch:+.2f} dRoll:{d_roll:+.2f}"
        cv2.putText(frame, info1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
        cv2.putText(frame, info2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

        cv2.imshow("Stable Tracker (moving anchor)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# ---------------- Cleanup -----------------------
cap.release()
cv2.destroyAllWindows()
print("Stopped.")
