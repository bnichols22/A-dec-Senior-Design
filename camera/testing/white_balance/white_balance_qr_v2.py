import cv2
import numpy as np
import os
import time

BASE_DIR = os.path.expanduser("~/senior_design/A-dec-Senior-Design/camera/testing")
POSTER_CAPTURE_DIR = os.path.join(BASE_DIR, "poster_captures")
os.makedirs(POSTER_CAPTURE_DIR, exist_ok=True)

def save_poster_capture(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1.0) * 1000)
    capture_path = os.path.join(POSTER_CAPTURE_DIR, f"white_balance_qr_capture_{timestamp}_{milliseconds:03d}.jpg")
    if cv2.imwrite(capture_path, frame):
        print(f"Saved white balance QR capture: {capture_path}")
        return True

    print(f"Failed to save white balance QR capture: {capture_path}")
    return False

def get_color_imbalance(roi):
    """
    Analyze the QR code to find the color imbalance in the white areas.
    Returns the mean Blue and mean Red values.
    """
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Use Otsu's thresholding to isolate the white paper
    _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate the average color of the "White" pixels
    mean_b = np.mean(roi[:, :, 0][mask == 255])
    mean_r = np.mean(roi[:, :, 2][mask == 255])
    
    return mean_b, mean_r

# --- Main Loop ---

# NOTE: If hardware settings do not apply on Windows, try forcing DirectShow:
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap = cv2.VideoCapture(0)

# 1. Disable Auto White Balance (0.0 disables, 1.0 enables)
cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)

# 2. Read the camera's current WB temperature, or set a safe default (e.g., 4500K)
current_temp = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
if current_temp <= 0 or current_temp == -1:
    current_temp = 4500.0

# Apply the initial temperature to the camera
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, current_temp)

qr_detector = cv2.QRCodeDetector()
calibrated = False

print("Press and HOLD 'c' when the QR code is in frame to auto-tune hardware WB.")
print("Press 'p' to save a poster capture.")
print("Press 'r' to reset to a default temperature (4500K).")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Since we are modifying the hardware settings directly, 
    # the raw frame pulled from the camera is already balanced!
    display_frame = frame.copy()

    # Detect QR Code
    ret_qr, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

    if ret_qr:
        points = points.astype(int)
        for pts in points:
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            
            if not calibrated:
                cv2.putText(display_frame, "QR Detected! Hold 'c' to Tune", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display camera status
    status_color = (0, 255, 0) if calibrated else (0, 165, 255)
    cv2.putText(display_frame, f"Hardware Temp: {current_temp}K", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow('Hardware White Balance', display_frame)

    # Handle User Input
    key = cv2.waitKey(1) & 0xFF

    # Notice we check for 'c' and run a feedback loop
    if key == ord('c') and ret_qr:
        pts = points[0]
        x, y, w, h = cv2.boundingRect(pts)
        
        # Ensure ROI is within bounds
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            roi = frame[y:y+h, x:x+w]
            
            try:
                mean_b, mean_r = get_color_imbalance(roi)
                
                # --- Hardware Calibration Feedback Loop ---
                step = 100       # How much to adjust the Kelvin temperature per loop
                tolerance = 2.0  # Allowable difference between R and B values
                
                diff = mean_r - mean_b
                
                if abs(diff) > tolerance:
                    # If image is too red, tell the camera the light is warmer (lower Kelvin)
                    # If image is too blue, tell the camera the light is cooler (higher Kelvin)
                    if mean_r > mean_b:
                        current_temp -= step
                    else:
                        current_temp += step
                        
                    # Clamp temperature to typical webcam bounds (2000K to 10000K)
                    current_temp = max(2000.0, min(10000.0, current_temp))
                    
                    # Apply directly to the camera hardware
                    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, current_temp)
                    print(f"Tuning... New Temp: {current_temp}K | R: {mean_r:.1f}, B: {mean_b:.1f}")
                    calibrated = False # Un-lock if we are still tuning
                else:
                    calibrated = True
                    print(f"Calibration locked at {current_temp}K!")
                    
            except Exception as e:
                print(f"Calibration failed: {e}")

    elif key == ord('r'):
        current_temp = 4500.0
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, current_temp)
        calibrated = False
        print("Reset to default 4500K.")

    elif key == ord('p'):
        save_poster_capture(display_frame)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
