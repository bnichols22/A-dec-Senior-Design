# Senior Design Raspberry Pi Setup & Face Tracking

This document summarizes the setup steps, file organization, and scripts used for the **Senior Design face tracking project**. It serves as a reference for the project team when working on the Raspberry Pi.

---

## 1. General Terminal Setup & Fixes
- Cleaned up Raspberry Pi APT sources to fix repository warnings.  
- Installed missing dependencies for OpenCV GUI:
  ```bash
  sudo apt install -y libxcb-xinerama0
  ```
- Forced OpenCV/Qt to use **xcb** (avoids Wayland errors) by adding to `~/.bashrc`:
  ```bash
  export QT_QPA_PLATFORM=xcb
  ```
- Suppressed TensorFlow Lite & MediaPipe log spam by adding to scripts:
  ```python
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  ```
- Redirected noisy logs into a file instead of the terminal:
  ```python
  sys.stderr = open(LOG_PATH, "w")
  ```

---

## 2. Virtual Environment (venv)
- **Purpose**:  
  A `venv` is a Python sandbox that isolates project dependencies. This avoids conflicts with system Python and ensures reproducibility.

- **Setup**:
  ```bash
  python3 -m venv ~/senior_design/venv
  source ~/senior_design/venv/bin/activate
  pip install -r requirements.txt
  ```

- **Freezing dependencies** (to share with others):
  ```bash
  pip freeze > requirements.txt
  ```

---

## 3. File Structure
```
~/senior_design/
│
├── venv/                     # Python virtual environment
│
├── camera/
│   └── testing/
│       ├── camera_test.py     # basic camera test script
│       ├── face_track.py      # main face/mouth tracking script
│       └── face_track_log.txt # overwritten log file (stderr redirected here)
```

---

## 4. Log File Behavior
- Path:  
  ```
  ~/senior_design/camera/testing/face_track_log.txt
  ```
- Overwritten every run of `face_track.py`.  
- Captures all stderr messages (warnings, info, errors).  

- Example to view logs:
  ```bash
  cat ~/senior_design/camera/testing/face_track_log.txt
  ```

- To watch live updates:
  ```bash
  tail -f ~/senior_design/camera/testing/face_track_log.txt
  ```

---

## 5. Working With Files on the Pi
- Open/edit with nano (terminal):
  ```bash
  nano ~/senior_design/camera/testing/face_track.py
  ```

- Open/edit with GUI editor:
  ```bash
  mousepad ~/senior_design/camera/testing/face_track.py
  ```

- Run the script:
  ```bash
  python ~/senior_design/camera/testing/face_track.py
  ```

---

## 6. What `face_track.py` Does
- **Purpose**: Detects and tracks a patient’s mouth using computer vision.  

- **Libraries Used**:
  - `OpenCV (cv2)` → camera input & drawing  
  - `MediaPipe FaceMesh` → pre-trained face landmark detection  
  - `collections.deque`, `time` → FPS calculation  
  - `os`, `sys`, `warnings` → environment setup, logging, suppression of noise  

- **Process**:
  1. Capture frames from the ELP 4K USB camera.  
  2. Detect face landmarks with MediaPipe.  
  3. Extract mouth landmarks, compute center point.  
  4. Apply Exponential Moving Average (EMA) smoothing to reduce jitter.  
  5. Draw crosshair on the smoothed point.  
  6. Display FPS and confidence overlay.  
  7. Log library messages into `face_track_log.txt`.  

- **Role in project**: This script forms the **vision backbone** of the dental light-tracking system. The smoothed mouth coordinates can later be used to drive a gimbal/light system.

---

## 7. Virtual Environment Management on New Devices

### Reproducing venv on a new Raspberry Pi
1. Copy the project folder (including `requirements.txt`) to the new Pi.  
2. Recreate venv:
   ```bash
   python3 -m venv ~/senior_design/venv
   source ~/senior_design/venv/bin/activate
   pip install -r requirements.txt
   ```

### Auto-activation on terminal open
We added this line to `~/.bashrc` so the venv activates automatically:
```bash
source ~/senior_design/venv/bin/activate
```

### Disabling venv
If you want to temporarily leave the venv:
```bash
deactivate
```

### Removing venv completely
To delete the environment:
```bash
rm -rf ~/senior_design/venv
```

---

# Recap
- We set up the Pi environment, fixed warnings, and installed dependencies.  
- Built and configured a **virtual environment** that auto-activates.  
- Organized files for testing, tracking, and logging.  
- Created `face_track.py` to track mouths using MediaPipe & OpenCV.  
- Redirected logs into a file (`face_track_log.txt`) to keep the terminal clean.  
- Documented how to reproduce venvs, disable/remove them, and share dependencies across Pis.  

---
