import cv2
import numpy as np
import json
import os

# --- Configuration ---


# Default state used if no profile is loaded
settings = {
    "exposure": -2,       # Example -2 (bright) -11(dark)
    "brightness": 0,      # Example -130 (dark) +130(bright)
    "contrast": 130,      # Example -130 (dark) +130(bright)
    "focus": 0,           # Example 0 - 500
    "white_balance": 4600,# min=2800 max=6500 step=1 default=4600
    "camera_id": 0        # 0 to N
}

live_feed = True

def save_profile(current_settings, filename):
    """Writes the current settings dictionary to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(current_settings, f, indent=4)
        print(f"--> Profile SAVED to {filename}")
    except Exception as e:
        print(f"Error saving profile: {e}")

def load_profile(filename):
    """Reads the JSON file and returns the dictionary."""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return None
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"--> Profile LOADED from {filename}")
        return data
    except Exception as e:
        print(f"Error loading profile: {e}")
        return None

def apply_camera_settings(cap, current_settings):
    """Applies all values in the settings dict to the OpenCV capture object."""
    if not cap.isOpened(): 
        return

    # Note: Different cameras support different properties. 
    # If a camera doesn't support a property, cv2 usually just ignores it or returns false.
    cap.set(cv2.CAP_PROP_EXPOSURE, current_settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, current_settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, current_settings["contrast"])
    
    # Disable autofocus to allow manual focus setting
    #cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
    #cap.set(cv2.CAP_PROP_FOCUS, current_settings["focus"])
    
    #cap.set(cv2.CAP_PROP_WB_TEMPERATURE, current_settings["white_balance"])


# --- Main Initialization ---
vid = cv2.VideoCapture(settings["camera_id"])
if not vid.isOpened():
    raise ValueError('Unable to open video source')

# Apply the default settings immediately upon open
apply_camera_settings(vid, settings)

blank_image = np.zeros((200,200,3), np.uint8)

print("Press the following key (lowercase or caps-lock) to change the setting:")
print("1,2,3: Switch to another webcam")
print("c/x  : decrease/increase Contrast")
print("b/v  : decrease/increase Brightness")
print("f/d  : decrease/increase Focus")
print("e/w  : decrease/increase Exposure")
print("u/y  : decrease/increase White Balance")
print("l/k  : hide/show live stream")
print(" s   : open DirectShow settings")
print(" p   : SAVE current profile to JSON") 
print(" o   : LOAD profile from JSON")
print(" q   : exit the application")

vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
vid.set(cv2.CAP_PROP_GAIN,0)

while(True):
    if live_feed:
        _, frame = vid.read()
        if frame is not None:
            cv2.imshow('image',frame)
    else:
        cv2.imshow('image',blank_image)
        frame = None
        
    key = cv2.waitKey(10)
    
    # EXIT
    if key == ord('q') or key == ord('Q'):
        break
    
    # SAVE PROFILE
    if key == ord('p') or key == ord('P'):
        profile_write = input("Enter profile file name to write to: ")
        save_profile(settings, profile_write)

    # LOAD PROFILE
    if key == ord('o') or key == ord('O'):
        profile_to_load = input("Enter name of profile to load: ")
        new_settings = load_profile(profile_to_load)
        if new_settings:
            settings = new_settings
            # If the loaded profile has a different camera ID, we might want to switch
            # For now, we will just apply image settings to the CURRENT camera
            apply_camera_settings(vid, settings)
            print(f"Loaded: {settings}")

    # DIRECTSHOW
    if key == ord('s') or key == ord('S'):
        print("Open DirectShow settings")
        vid.release()
        vid2 = cv2.VideoCapture(settings["camera_id"] + cv2.CAP_DSHOW)
        vid2.set(cv2.CAP_PROP_SETTINGS, 1)
        vid2.release()
        vid = cv2.VideoCapture(settings["camera_id"])
        apply_camera_settings(vid, settings) # Re-apply after reopening

    # HIDE LIVE
    if key == ord('l'):
        print(f'hide live video camera')
        vid.release()
        vid = cv2.VideoCapture(settings["camera_id"])
        apply_camera_settings(vid, settings)
        live_feed=False

    # SHOW LIVE
    if key == ord('k'):
        print(f'show live video camera (blocked for other processes)')
        if vid.isOpened():
            vid.release()
        vid = cv2.VideoCapture(settings["camera_id"])
        apply_camera_settings(vid, settings)
        live_feed=True

    # EXPOSURE
    if key == ord('w'):
        settings["exposure"]+=0.5
        r=vid.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
        print(f'exposure: {settings["exposure"]}')
    if key == ord('e'):
        settings["exposure"]-=0.5
        print(f'exposure: {settings["exposure"]}')
        r=vid.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])

    # WHITE BALANCE
    if key == ord('u'):
        settings["white_balance"] += 100
        r=vid.set(cv2.CAP_PROP_WB_TEMPERATURE, settings["white_balance"])
        print(f'white balance: {settings["white_balance"]}')
    if key == ord('y'):
        settings["white_balance"] -= 100
        r=vid.set(cv2.CAP_PROP_WB_TEMPERATURE, settings["white_balance"])
        print(f'white balance: {settings["white_balance"]}')

    # BRIGHTNESS
    if key == ord('v'):
        settings["brightness"]+=10
        print(f'brightness: {settings["brightness"]}')
        r=vid.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    if key == ord('b'):
        settings["brightness"]-=10
        print(f'brightness: {settings["brightness"]}')
        vid.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])

    # CONTRAST
    if key == ord('x'):
        settings["contrast"]+=10
        print(f'contrast: {settings["contrast"]}')
        vid.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    if key == ord('c'):
        settings["contrast"]-=10
        print(f'contrast: {settings["contrast"]}')
        vid.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])

    # FOCUS
    if key == ord('d'):
        settings["focus"]+=5
        print(f'focus: {settings["focus"]}')
        vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        vid.set(cv2.CAP_PROP_FOCUS, settings["focus"])
    if key == ord('f'):
        settings["focus"]-=5
        print(f'focus: {settings["focus"]}')
        vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        vid.set(cv2.CAP_PROP_FOCUS, settings["focus"])

    # CAMERA SWITCH
    if key >= ord('0') and key <= ord('3'):
        vid.release()
        settings["camera_id"] = key-ord('0')
        vid = cv2.VideoCapture(settings["camera_id"])
        if not vid.isOpened():
            raise ValueError('Unable to open video source')
        apply_camera_settings(vid, settings)
        
        
if vid.isOpened():
    vid.release()
cv2.destroyAllWindows()