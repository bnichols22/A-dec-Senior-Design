import cv2
import numpy as np

def apply_white_balance(frame, b_gain, g_gain, r_gain):
    """
    Apply calculated gains to the frame.
    """
    # Merge the channels with the gains applied
    # We use np.clip to ensure values don't exceed 255
    b, g, r = cv2.split(frame)
    b = cv2.multiply(b, b_gain)
    g = cv2.multiply(g, g_gain)
    r = cv2.multiply(r, r_gain)
    
    balanced_frame = cv2.merge([b, g, r])
    balanced_frame = np.clip(balanced_frame, 0, 255).astype(np.uint8)
    return balanced_frame

def get_white_balance_gains(roi):
    """
    Analyze a region of interest (ROI) to find the color imbalance.
    We assume the ROI is the QR code (Black and White).
    """
    # Convert to grayscale to easily separate the black module form the white background
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding to separate Black (QR data) from White (Paper)
    # This creates a mask where '255' represents the white parts of the paper
    _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate the average color of the "White" pixels only
    mean_b = np.mean(roi[:, :, 0][mask == 255])
    mean_g = np.mean(roi[:, :, 1][mask == 255])
    mean_r = np.mean(roi[:, :, 2][mask == 255])

    # Avoid division by zero
    if mean_b == 0: mean_b = 1
    if mean_g == 0: mean_g = 1
    if mean_r == 0: mean_r = 1

    # Standard "Gray World" assumption: Normalize to the Green channel 
    # (Green is usually the reference for sensors)
    g_gain = 1.0
    b_gain = mean_g / mean_b
    r_gain = mean_g / mean_r
    
    return b_gain, g_gain, r_gain

# --- Main Loop ---

cap = cv2.VideoCapture(0)
qr_detector = cv2.QRCodeDetector()

# Default gains (1.0 means no change)
current_b_gain, current_g_gain, current_r_gain = 1.0, 1.0, 1.0
calibrated = False

print("Press 'c' when the QR code is in frame to calibrate.")
print("Press 'r' to reset white balance.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Apply the current white balance settings to the frame
    # We do this immediately so you can see the result of your calibration
    display_frame = apply_white_balance(frame, current_b_gain, current_g_gain, current_r_gain)

    # 2. Detect QR Code
    # We detect on the *original* raw frame to ensure detection robustness, 
    # but you could also detect on the balanced frame.
    ret_qr, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

    if ret_qr:
        points = points.astype(int)
        for pts in points:
            # Draw a box around the QR code just for visual feedback
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            
            # --- Calibration Trigger ---
            # Using waitKey logic inside the loop to capture the 'c' press
            # Note: We check key presses at the end of the loop, but we set a flag here
            # For simplicity, let's just use a distinct "Calibration Mode" visual
            cv2.putText(display_frame, "QR Detected! Press 'c' to Calibrate", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 3. Handle User Input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and ret_qr:
        # Extract the Region of Interest (ROI) based on QR points
        # We take a bounding box around the detected points
        pts = points[0]
        x, y, w, h = cv2.boundingRect(pts)
        
        # Ensure ROI is within frame bounds
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            roi = frame[y:y+h, x:x+w]
            
            # Calculate new gains
            try:
                b, g, r = get_white_balance_gains(roi)
                current_b_gain, current_g_gain, current_r_gain = b, g, r
                calibrated = True
                print(f"Calibrated! Gains - B: {b:.2f}, G: {g:.2f}, R: {r:.2f}")
            except Exception as e:
                print(f"Calibration failed (ROI likely too small): {e}")

    elif key == ord('r'):
        current_b_gain, current_g_gain, current_r_gain = 1.0, 1.0, 1.0
        print("Reset to default.")
        calibrated = False

    elif key == ord('q'):
        break

    # Display status
    if calibrated:
        cv2.putText(display_frame, f"WB Active (B:{current_b_gain:.1f} R:{current_r_gain:.1f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

    cv2.imshow('White Balance', display_frame)

cap.release()
cv2.destroyAllWindows()