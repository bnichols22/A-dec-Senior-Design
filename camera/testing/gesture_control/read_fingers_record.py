import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the landmark indices for tips and their corresponding pivot joints
FINGER_TIPS = [8, 12, 16, 20] 
THUMB_TIP = 4

cap = cv2.VideoCapture(0)

# --- RECORDING SETUP START ---
# Get the width and height of the video frame directly from the camera
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
# 'mp4v' is a standard codec for .mp4 files on most systems. 
# You can try 'avc1' if 'mp4v' fails on your specific OS.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
# --- RECORDING SETUP END ---

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Performance optimization: Mark as not writeable
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw annotations
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        finger_count = 0
        fingers_up_list = []

        if results.multi_hand_landmarks:
            # Handle potentially multiple hands
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # 1. Get Hand Label (Left vs Right) for Thumb Logic
                # Note: Because we flipped the image, MediaPipe's "Left" might appear as "Right" visually
                # depending on exact mirroring logic, but the label comes from the classification.
                if results.multi_handedness:
                    hand_label = results.multi_handedness[hand_idx].classification[0].label
                else:
                    hand_label = "Unknown"
                
                # List to store status of this hand's fingers
                hand_fingers = []

                # --- THUMB LOGIC ---
                thumb_tip_x = hand_landmarks.landmark[4].x
                thumb_ip_x = hand_landmarks.landmark[3].x
                
                # Depending on Left/Right hand, the "open" direction is different
                # Note: The original logic in your script assumes specific mirroring
                if hand_label == "Left": 
                    if thumb_tip_x > thumb_ip_x:
                        hand_fingers.append("Thumb")
                else: 
                    if thumb_tip_x < thumb_ip_x:
                        hand_fingers.append("Thumb")

                # --- FINGER LOGIC ---
                finger_names = ["Index", "Middle", "Ring", "Pinky"]
                
                for i, tip_idx in enumerate(FINGER_TIPS):
                    tip_y = hand_landmarks.landmark[tip_idx].y
                    pip_y = hand_landmarks.landmark[tip_idx - 2].y
                    
                    if tip_y < pip_y:
                        hand_fingers.append(finger_names[i])

                # Draw the standard landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # --- DISPLAY TEXT ---
                text_y = 50 + (hand_idx * 50)
                fingers_str = ", ".join(hand_fingers)
                
                cv2.putText(image, 
                            f"{hand_label}: {fingers_str}", 
                            (10, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2)

        # --- WRITE FRAME TO FILE ---
        # We write the image AFTER all drawings are complete
        out.write(image)

        cv2.imshow('MediaPipe Hands', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release everything when the job is finished
cap.release()
out.release() # Save the video file
cv2.destroyAllWindows()