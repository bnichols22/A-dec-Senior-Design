import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the landmark indices for tips and their corresponding pivot joints
# [Tip_Index, Pivot_Index]
# Pivot for fingers is usually the PIP joint (2 joints below tip)
# Pivot for thumb is usually the IP joint (1 joint below tip) or MCP
FINGER_TIPS = [8, 12, 16, 20] 
THUMB_TIP = 4

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

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
                # MediaPipe assumes input is NOT mirrored, but selfie cameras ARE mirrored.
                # We need the classification to know which way the thumb moves.
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                
                # List to store status of this hand's fingers (0=closed, 1=open)
                hand_fingers = []

                # --- THUMB LOGIC ---
                # Compare Thumb Tip (4) x-coord with IP Joint (3) x-coord
                # Note: This logic assumes the hand is upright.
                thumb_tip_x = hand_landmarks.landmark[4].x
                thumb_ip_x = hand_landmarks.landmark[3].x
                
                # Depending on Left/Right hand, the "open" direction is different
                if hand_label == "Left": 
                    # For a Left hand, thumb opens to the RIGHT side of the hand (in the image)
                    if thumb_tip_x > thumb_ip_x:
                        hand_fingers.append("Thumb")
                else: # Right Hand
                    # For a Right hand, thumb opens to the LEFT side of the hand (in the image)
                    if thumb_tip_x < thumb_ip_x:
                        hand_fingers.append("Thumb")

                # --- FINGER LOGIC (Index, Middle, Ring, Pinky) ---
                # Compare Tip y-coord with PIP Joint y-coord
                # Remember: Y-coordinates increase as you go DOWN the screen.
                # So, Tip < Pivot means the finger is UP.
                
                # Names for display
                finger_names = ["Index", "Middle", "Ring", "Pinky"]
                
                for i, tip_idx in enumerate(FINGER_TIPS):
                    # tip_idx is the tip, tip_idx-2 is the PIP joint
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
                # Calculate position for text based on hand bounding box or fixed
                text_y = 50 + (hand_idx * 50)
                fingers_str = ", ".join(hand_fingers)
                
                cv2.putText(image, 
                            f"{hand_label}: {fingers_str}", 
                            (10, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2)

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()