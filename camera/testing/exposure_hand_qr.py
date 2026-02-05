import cv2
import mediapipe as mp
import numpy as np
import json
import time

# --- CONFIGURATION & DEFAULTS ---

# Default Software Gains ("Colder" look: Higher Blue, Lower Red)
# Adjust these values to change the default "Reset" temperature.
DEFAULT_WB_GAINS = {"b": 1.15, "g": 1.0, "r": 0.90}

param_ranges = {
    "exposure":      {"min": -10,  "max": -1,   "default": -5},
    "brightness":    {"min": 0,    "max": 255,  "default": 128},
    "contrast":      {"min": 0,    "max": 255,  "default": 128}
}

actions_to_map = ["exposure", "brightness", "contrast", "SAVE_PROFILE"]

# --- GLOBAL STATE ---
current_settings = {k: v["default"] for k, v in param_ranges.items()}
wb_gains = DEFAULT_WB_GAINS.copy()
gesture_map = {} 
calibration_index = 0
is_calibrated = False
last_save_time = 0

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- HELPER FUNCTIONS ---

def apply_white_balance(frame, b_gain, g_gain, r_gain):
    """Apply calculated gains to the frame (Software White Balance)."""
    b, g, r = cv2.split(frame)
    b = cv2.multiply(b, b_gain)
    g = cv2.multiply(g, g_gain)
    r = cv2.multiply(r, r_gain)
    balanced_frame = cv2.merge([b, g, r])
    balanced_frame = np.clip(balanced_frame, 0, 255).astype(np.uint8)
    return balanced_frame

def get_white_balance_gains(roi):
    """Analyze ROI (QR Code) to find color imbalance."""
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    mean_b = np.mean(roi[:, :, 0][mask == 255])
    mean_g = np.mean(roi[:, :, 1][mask == 255])
    mean_r = np.mean(roi[:, :, 2][mask == 255])

    if mean_b == 0: mean_b = 1
    if mean_g == 0: mean_g = 1
    if mean_r == 0: mean_r = 1

    # Standard "Gray World" assumption
    g_gain = 1.0
    b_gain = mean_g / mean_b
    r_gain = mean_g / mean_r
    
    return b_gain, g_gain, r_gain

def save_profile(settings, gains, filename="camera_profile.json"):
    full_profile = {"hardware_settings": settings, "software_gains": gains}
    try:
        with open(filename, 'w') as f:
            json.dump(full_profile, f, indent=4)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def apply_hardware_settings(cap, settings):
    if not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    # Lock Hardware WB to a fixed neutral-ish value so software can do the rest
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600) 

def get_fingers_status(hand_landmarks, handedness_label):
    fingers = []
    # Thumb
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x
    is_thumb_open = False
    if handedness_label == "Left": 
        if thumb_tip_x > thumb_ip_x: is_thumb_open = True
    else: 
        if thumb_tip_x < thumb_ip_x: is_thumb_open = True
    fingers.append(1 if is_thumb_open else 0)

    # Fingers
    tips = [8, 12, 16, 20] 
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return tuple(fingers)

def map_y_to_value(y_norm, param_name):
    p_min = param_ranges[param_name]["min"]
    p_max = param_ranges[param_name]["max"]
    y_norm = max(0.0, min(1.0, y_norm))
    val = p_min + (1.0 - y_norm) * (p_max - p_min)
    if isinstance(p_min, int) and isinstance(p_max, int):
        return int(val)
    return val

# --- MAIN EXECUTION ---

def main():
    global is_calibrated, calibration_index, current_settings, last_save_time, wb_gains

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise ValueError("Cannot open camera")

    qr_detector = cv2.QRCodeDetector()
    apply_hardware_settings(cap, current_settings)

    print("--- Camera Controls ---")
    print("'r' : Reset White Balance to Cool Default")
    print("'c' : Calibrate White Balance (When QR is visible)")
    print("'m' : Map Gesture (When Mapping Mode is active)")
    print("'q' : Quit")

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1) as hands:

        while cap.isOpened():
            success, raw_frame = cap.read()
            if not success: continue

            # 1. Flip & Apply Software WB
            raw_frame = cv2.flip(raw_frame, 1)
            display_frame = apply_white_balance(raw_frame, wb_gains['b'], wb_gains['g'], wb_gains['r'])
            h, w, _ = display_frame.shape

            # 2. QR Detection (Run on RAW frame for accuracy)
            ret_qr, _, points, _ = qr_detector.detectAndDecodeMulti(raw_frame)
            
            # --- BRANCHING LOGIC ---
            
            if ret_qr:
                # === STATE A: QR DETECTED (GESTURES PAUSED) ===
                if points is not None:
                    points = points.astype(int)
                    for pts in points:
                        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                
                # Visual Feedback
                cv2.rectangle(display_frame, (0, 0), (w, 85), (0,0,0), -1)
                cv2.putText(display_frame, "QR DETECTED - GESTURES PAUSED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, "Press 'c' to Calibrate White Balance", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            else:
                # === STATE B: NO QR (GESTURES ACTIVE) ===
                display_frame.flags.writeable = False
                image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                display_frame.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        handedness = results.multi_handedness[hand_idx].classification[0].label
                        fingers_tuple = get_fingers_status(hand_landmarks, handedness)
                        hand_center_y = hand_landmarks.landmark[0].y

                        # Gesture Mapping Wizard
                        if not is_calibrated:
                            target_action = actions_to_map[calibration_index]
                            cv2.putText(display_frame, "MAPPING MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.putText(display_frame, f"Action: {target_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            cv2.putText(display_frame, f"Gesture: {fingers_tuple}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                            cv2.putText(display_frame, "Hold & Press 'm'", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Control Mode
                        else:
                            active_param = gesture_map.get(fingers_tuple)
                            if active_param:
                                if active_param == "SAVE_PROFILE":
                                    if time.time() - last_save_time > 3.0:
                                        if save_profile(current_settings, wb_gains):
                                            cv2.putText(display_frame, "SAVED!", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                                            last_save_time = time.time()
                                else:
                                    # Update Settings
                                    new_val = map_y_to_value(hand_center_y, active_param)
                                    current_settings[active_param] = new_val
                                    apply_hardware_settings(cap, current_settings)
                                    
                                    # UI
                                    cv2.putText(display_frame, f"{active_param.upper()}: {new_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                    # Slider
                                    bar_x, bar_h, bar_y = 50, 300, (h-300)//2
                                    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x+20, bar_y+bar_h), (255,255,255), 1)
                                    ind_y = max(bar_y, min(bar_y+bar_h, int(bar_y + (hand_center_y * bar_h))))
                                    cv2.circle(display_frame, (bar_x+10, ind_y), 10, (0, 255, 0), -1)
                            else:
                                cv2.putText(display_frame, "Ready", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                else:
                    if not is_calibrated:
                         cv2.putText(display_frame, "Show Hand to Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

            # --- INPUT HANDLING ---
            key = cv2.waitKey(5) & 0xFF
            
            # 'c': Calibrate WB (Only if QR is visible)
            if key == ord('c') and ret_qr:
                if points is not None:
                    pts = points[0]
                    x, y, w_rect, h_rect = cv2.boundingRect(pts)
                    if x >= 0 and y >= 0 and w_rect > 0 and h_rect > 0:
                        roi = raw_frame[y:y+h_rect, x:x+w_rect]
                        try:
                            b, g, r = get_white_balance_gains(roi)
                            wb_gains = {"b": b, "g": g, "r": r}
                            print(f"WB Calibrated! B:{b:.2f} G:{g:.2f} R:{r:.2f}")
                        except Exception as e: print(e)

            # 'm': Map Gesture (Only if QR NOT visible)
            if key == ord('m') and not ret_qr and not is_calibrated and results.multi_hand_landmarks:
                if fingers_tuple not in gesture_map:
                    gesture_map[fingers_tuple] = actions_to_map[calibration_index]
                    calibration_index += 1
                    if calibration_index >= len(actions_to_map): is_calibrated = True

            # 'r': Reset WB to Default (Cold)
            if key == ord('r'): 
                wb_gains = DEFAULT_WB_GAINS.copy()
                print("WB Reset to Cold Defaults")
                
            if key == 27 or key == ord('q'): break

            # --- INFO OVERLAY ---
            if is_calibrated and not ret_qr:
                y_off = h - 140
                for k, v in current_settings.items():
                    cv2.putText(display_frame, f"{k}: {v}", (w - 200, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    y_off += 20
                cv2.putText(display_frame, f"WB: B:{wb_gains['b']:.2f} R:{wb_gains['r']:.2f}", (w - 200, y_off+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

            cv2.imshow('Camera Control', display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()