

---

## 🧩 What `centroid_track.py` Does

- Detects mouth landmarks (13, 14, 61, 291) via **MediaPipe FaceMesh**.
- Computes the **centroid** of those landmarks and smooths it via **Exponential Moving Average (EMA)**.
- Defines a **stable box (neutral zone)** centered around the last stable centroid.
- Converts pixel deltas to angular deltas (`Δyaw`, `Δpitch`, `Δroll`) based on camera FOV.
- Emits **one motion command** when the centroid leaves the box and the **rate limiter** allows it.
- Waits for stability (low velocity + low jitter) before **re-anchoring** the box to the new centroid.
- Logs and displays telemetry for every frame.

This approach results in a quiet, stable tracking system — no chatter, no constant small adjustments — ideal for a gimbal that needs deliberate control.

---

## 🧾 Console Telemetry

Every frame prints a line like:

t_s d_yaw d_pitch d_roll | anchor_x anchor_y stable can_send


| Field | Description |
|--------|-------------|
| `t_s` | Seconds since program start |
| `d_yaw`, `d_pitch`, `d_roll` | Angular deltas (in °); `0.000` means no correction needed |
| `anchor_x`, `anchor_y` | Pixel location of the stable box’s center (anchor) |
| `stable` | `1` if the centroid is calm and stable |
| `can_send` | `1` if a command can be sent (rate limit passed) |

**Interpretation:**
- If the centroid is inside the stable box → all deltas = 0.  
- If outside and rate-limited gate open → nonzero deltas (yaw, pitch, possibly roll).  
- After stability, the box re-anchors to the new location.

**Sign conventions**
- +Yaw → face right → gimbal right (reverse if needed)
- +Pitch → face down → gimbal down
- +Roll → right corner lower than left (noisy; thresholded separately)

---

## ⚙️ Stability & Command Logic

1. **Initialize:** First detection sets the initial anchor.
2. **Stable Zone:** A neutral box forms around the anchor.
3. **No Output:** If centroid stays inside → no command.
4. **Emit Once:** When centroid exits and allowed → one `(Δyaw, Δpitch, Δroll)` is sent.
5. **Re-anchor:** After stable dwell period → move the anchor to new centroid.
6. **Repeat.**

This ensures smooth transitions and avoids unnecessary gimbal motion.

---

## 🔧 Adjustable Parameters

Edit these constants at the top of `centroid_track.py`:

| Parameter | Description | Typical Range |
|------------|--------------|----------------|
| `FOV_H_DEG`, `FOV_V_DEG` | Camera FOV for degree mapping | 95 × 60 |
| `NEUTRAL_BOX_NORM_W/H` | Stable box half-size (fraction of frame) | 0.04–0.08 |
| `MIN_SEND_INTERVAL_S` | Seconds between sends | 1.0–3.0 |
| `MIN_STEP_DEG` | Ignore tiny yaw/pitch changes | 0.5–1.2 |
| `ROLL_MIN_STEP_DEG` | Ignore tiny roll changes | 1.0–2.0 |
| `VEL_THRESH_DEG_S` | Angular speed limit for stability | 2–3 |
| `POS_STD_THRESH_PX` | Pixel jitter tolerance | 2–5 |
| `STABLE_DWELL_S` | Time to remain calm before re-anchoring | 0.3–1.0 |
| `SMOOTH_ALPHA` | EMA smoothing factor | 0.2–0.3 |

**Tuning tips**
- Smaller neutral box = more responsive, more motion.
- Longer send interval = less frequent movement.
- Higher thresholds = smoother, slower reaction.

---

## 🧮 Data Flow Summary

1. **Frame Capture** → OpenCV grabs frames.  
2. **Landmark Detection** → MediaPipe finds mouth points.  
3. **Centroid Calculation** → Average + smooth position.  
4. **Angular Conversion** → Convert pixel deltas to degrees.  
5. **Stability Window** → Check stddev & velocity history.  
6. **Decision Gate** → If outside stable box + rate limit passed → emit delta.  
7. **Re-anchor** → After calm dwell, reset stable region center.  
8. **Repeat** continuously.
