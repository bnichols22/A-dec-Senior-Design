import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# --- CONFIGURATION & RANGES ---
# Define the min/max ranges for your specific camera. 
# Note: These vary wildly between webcams. You may need to adjust "min" and "max".
param_ranges = {
    "exposure":      {"min": -10,  "max": -1,   "default": -5},
    "brightness":    {"min": 0,    "max": 255,  "default": 128},
    "contrast":      {"min": 0,    "max": 255,  "default": 128},
    "focus":         {"min": 0,    "max": 255,  "default": 0, "manual": True}, # manual=True disables autofocus
    "white_balance": {"min": 2800, "max": 6500, "default": 4600}
}

# The actions we want to map gestures to
actions_to_map = ["exposure", "brightness", "contrast", "focus", "white_balance", "SAVE_PROFILE"]

# --- GLOBAL STATE ---
current_settings = {k: v["default"] for k, v in param_ranges.items()}
gesture_map = {}  # Format: { (1,0,0,0,0): "exposure", ... }
calibration_index = 0
is_calibrated = False
last_save_time = 0

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- HELPER FUNCTIONS ---

def save_profile(settings, filename="camera_profile.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving profile: {e}")
        return False

def apply_camera_settings(cap, settings):
    """Applies the settings dictionary to the OpenCV Capture object."""
    if not cap.isOpened(): return

    # EXPOSURE
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    
    # BRIGHTNESS
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    
    # CONTRAST
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    
    # FOCUS (Requires disabling autofocus first)
    if param_ranges["focus"]["manual"]:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, settings["focus"])
    
    # WHITE BALANCE
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settings["white_balance"])

def get_fingers_status(hand_landmarks, handedness_label):
    """
    Returns a tuple of 5 booleans representing fingers: (Thumb, Index, Middle, Ring, Pinky)
    1 = Open/Up, 0 = Closed/Down
    """
    fingers = []
    
    # --- THUMB LOGIC ---
    # Thumb tip (4) vs IP joint (3). Direction depends on Left/Right hand.
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x
    
    is_thumb_open = False
    if handedness_label == "Left": # Left hand (in selfie view, assumes mirrored logic handled by caller?)
        # Note: MediaPipe output is not mirrored by default unless image is flipped.
        # If we flip image BEFORE processing, Left is actually Left on screen.
        if thumb_tip_x > thumb_ip_x: is_thumb_open = True
    else: # Right Hand
        if thumb_tip_x < thumb_ip_x: is_thumb_open = True
    
    fingers.append(1 if is_thumb_open else 0)

    # --- FINGERS LOGIC ---
    # Tip y < PIP y means finger is UP (because Y increases downwards)
    tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return tuple(fingers)

def map_y_to_value(y_norm, param_name):
    """
    Maps normalized Y coordinate (0.0 top to 1.0 bottom) to the parameter range.
    Inverting Y so 0.0 (top) is MAX value and 1.0 (bottom) is MIN value for intuitive UI.
    """
    p_min = param_ranges[param_name]["min"]
    p_max = param_ranges[param_name]["max"]
    
    # Clamp Y
    y_norm = max(0.0, min(1.0, y_norm))
    
    # Inverse mapping: Top of screen (0) -> Max Value
    val = p_min + (1.0 - y_norm) * (p_max - p_min)
    
    # Rounding logic
    if isinstance(p_min, int) and isinstance(p_max, int):
        return int(val)
    return val

# --- MAIN EXECUTION ---

def main():
    global is_calibrated, calibration_index, current_settings, last_save_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open camera")

    # Apply defaults initially
    apply_camera_settings(cap, current_settings)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            # Flip for selfie view (mirror effect)
            image = cv2.flip(image, 1)
            h, w, _ = image.shape

            # Process Hands
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            # Defaults
            active_gesture = None
            hand_center_y = 0.5 # Normalized 0 to 1
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Get Handedness
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                    
                    # Get Finger State (The Gesture)
                    fingers_tuple = get_fingers_status(hand_landmarks, handedness)
                    
                    # Get Hand Vertical Position (using Wrist landmark 0 as reference)
                    hand_center_y = hand_landmarks.landmark[0].y

                    # --- PHASE 1: CALIBRATION WIZARD ---
                    if not is_calibrated:
                        target_action = actions_to_map[calibration_index]
                        
                        # UI Text
                        cv2.putText(image, "CALIBRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(image, f"Step {calibration_index + 1}/{len(actions_to_map)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                        cv2.putText(image, f"Action: {target_action.upper()}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.putText(image, f"Detected: {fingers_tuple}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        cv2.putText(image, "Hold gesture & Press 'c' to map", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        key = cv2.waitKey(5)
                        if key == ord('c') or key == ord('C'):
                            # Check if gesture is already used
                            if fingers_tuple in gesture_map:
                                print(f"Gesture {fingers_tuple} already mapped to {gesture_map[fingers_tuple]}!")
                            else:
                                gesture_map[fingers_tuple] = target_action
                                print(f"Mapped {fingers_tuple} to {target_action}")
                                calibration_index += 1
                                if calibration_index >= len(actions_to_map):
                                    is_calibrated = True
                                    print("Calibration Complete!")
                                    print(gesture_map)
                                    time.sleep(0.5) 

                    # --- PHASE 2: CONTROL MODE ---
                    else:
                        active_param = gesture_map.get(fingers_tuple)

                        # Logic: If gesture is mapped to a parameter, adjust it based on Y height
                        if active_param:
                            if active_param == "SAVE_PROFILE":
                                # Debounce save
                                if time.time() - last_save_time > 3.0:
                                    if save_profile(current_settings):
                                        cv2.putText(image, "SAVED!", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                                        last_save_time = time.time()
                                else:
                                     cv2.putText(image, "Saving...", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            else:
                                # It's a camera setting (Exposure, Brightness, etc.)
                                new_val = map_y_to_value(hand_center_y, active_param)
                                current_settings[active_param] = new_val
                                apply_camera_settings(cap, current_settings)
                                
                                # Visual Feedback Bars
                                cv2.putText(image, f"Setting: {active_param.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                cv2.putText(image, f"Val: {new_val}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                
                                # Draw a slider bar on the side
                                bar_x = 50
                                bar_h = 300
                                bar_y_start = (h - bar_h) // 2
                                cv2.rectangle(image, (bar_x, bar_y_start), (bar_x+20, bar_y_start+bar_h), (255,255,255), 1)
                                # Indicator
                                ind_y = int(bar_y_start + (hand_center_y * bar_h))
                                ind_y = max(bar_y_start, min(bar_y_start+bar_h, ind_y))
                                cv2.circle(image, (bar_x+10, ind_y), 10, (0, 255, 0), -1)
                        
                        else:
                            # Idle state
                            cv2.putText(image, "Control Mode: Ready", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                            cv2.putText(image, "Show gesture to adjust.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- RENDER CURRENT SETTINGS OVERLAY ---
            if is_calibrated:
                y_off = h - 140
                for k, v in current_settings.items():
                    cv2.putText(image, f"{k}: {v}", (w - 200, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    y_off += 20

            cv2.imshow('Gesture Camera Controller', image)

            if cv2.waitKey(5) & 0xFF == 27: # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()