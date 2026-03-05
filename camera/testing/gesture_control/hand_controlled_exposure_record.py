import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# --- CONFIGURATION & RANGES ---
param_ranges = {
    "exposure":      {"min": -10,  "max": -1,   "default": -5},
    "brightness":    {"min": 0,    "max": 255,  "default": 128},
    "contrast":      {"min": 0,    "max": 255,  "default": 128},
    "focus":         {"min": 0,    "max": 255,  "default": 0, "manual": True},
    "white_balance": {"min": 2800, "max": 6500, "default": 4600}
}

actions_to_map = ["exposure", "brightness", "contrast", "focus", "white_balance", "SAVE_PROFILE"]

# --- GLOBAL STATE ---
current_settings = {k: v["default"] for k, v in param_ranges.items()}
gesture_map = {}
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
    # Note: CAP_PROP_EXPOSURE behavior varies wildly by camera driver (v4l2 vs others)
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    
    # FOCUS
    if param_ranges["focus"].get("manual"):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, settings["focus"])
    
    # WHITE BALANCE
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settings["white_balance"])

def get_fingers_status(hand_landmarks, handedness_label):
    fingers = []
    
    # THUMB LOGIC
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x
    
    is_thumb_open = False
    if handedness_label == "Left": 
        if thumb_tip_x > thumb_ip_x: is_thumb_open = True
    else: 
        if thumb_tip_x < thumb_ip_x: is_thumb_open = True
    
    fingers.append(1 if is_thumb_open else 0)

    # FINGERS LOGIC
    tips = [8, 12, 16, 20] 
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return tuple(fingers)

def map_y_to_value(y_norm, param_name):
    """Maps normalized Y coordinate to parameter range."""
    p_min = param_ranges[param_name]["min"]
    p_max = param_ranges[param_name]["max"]
    
    # Clamp Y
    y_norm = max(0.0, min(1.0, y_norm))
    
    # Inverse mapping: Top (0) -> Max, Bottom (1) -> Min
    val = p_min + (1.0 - y_norm) * (p_max - p_min)
    
    if isinstance(p_min, int) and isinstance(p_max, int):
        return int(val)
    return val

# --- MAIN EXECUTION ---

def main():
    global is_calibrated, calibration_index, current_settings, last_save_time, gesture_map

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open camera")

    # --- RECORDING SETUP START ---
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Using XVID / .avi for Raspberry Pi compatibility
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Save to current directory
    out = cv2.VideoWriter('gesture_control_session.avi', fourcc, 20.0, (frame_width, frame_height))
    print(f"Recording started. Video will be saved to 'gesture_control_session.avi'")
    # --- RECORDING SETUP END ---

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

            # Flip for selfie view
            image = cv2.flip(image, 1)
            h, w, _ = image.shape

            # Process Hands
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            # Logic Variables
            active_gesture_tuple = None
            hand_center_y = 0.5 
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Get Handedness
                    if results.multi_handedness:
                        handedness = results.multi_handedness[hand_idx].classification[0].label
                    else:
                        handedness = "Unknown"
                    
                    # Get Finger State
                    fingers_tuple = get_fingers_status(hand_landmarks, handedness)
                    active_gesture_tuple = fingers_tuple

                    # Get Hand Vertical Position (Wrist)
                    hand_center_y = hand_landmarks.landmark[0].y

            # --- LOGIC HANDLING (Outside loop to ensure we draw UI even if no hand detected) ---
            
            # PHASE 1: CALIBRATION WIZARD
            if not is_calibrated:
                target_action = actions_to_map[calibration_index]
                
                cv2.putText(image, "CALIBRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Step {calibration_index + 1}/{len(actions_to_map)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(image, f"Action: {target_action.upper()}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                if active_gesture_tuple:
                    cv2.putText(image, f"Detected: {active_gesture_tuple}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(image, "Hold gesture & Press 'c' to map", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                key = cv2.waitKey(5)
                if key == ord('c') or key == ord('C'):
                    if active_gesture_tuple:
                        if active_gesture_tuple in gesture_map:
                            print(f"Gesture {active_gesture_tuple} already mapped to {gesture_map[active_gesture_tuple]}!")
                        else:
                            gesture_map[active_gesture_tuple] = target_action
                            print(f"Mapped {active_gesture_tuple} to {target_action}")
                            calibration_index += 1
                            if calibration_index >= len(actions_to_map):
                                is_calibrated = True
                                print("Calibration Complete!")
                                print(gesture_map)
                                time.sleep(0.5)

            # PHASE 2: CONTROL MODE
            else:
                active_param = None
                if active_gesture_tuple:
                    active_param = gesture_map.get(active_gesture_tuple)

                if active_param:
                    if active_param == "SAVE_PROFILE":
                        # Debounce save
                        if time.time() - last_save_time > 3.0:
                            save_profile(current_settings)
                            cv2.putText(image, "SAVED!", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                            last_save_time = time.time()
                        else:
                             cv2.putText(image, "Saved.", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    else:
                        # Map Y position to value
                        new_val = map_y_to_value(hand_center_y, active_param)
                        current_settings[active_param] = new_val
                        apply_camera_settings(cap, current_settings)
                        
                        # Visual Feedback
                        cv2.putText(image, f"Adjusting: {active_param.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(image, f"Val: {new_val}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Slider Bar
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

                # Render settings overlay (always visible in control mode)
                y_off = h - 140
                for k, v in current_settings.items():
                    cv2.putText(image, f"{k}: {v}", (w - 220, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_off += 20

            # --- WRITE FRAME TO FILE ---
            out.write(image)

            cv2.imshow('Gesture Camera Controller', image)

            if cv2.waitKey(5) & 0xFF == 27: # ESC to quit
                break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved successfully.")

if __name__ == "__main__":
    main()