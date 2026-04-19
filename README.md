# A-dec Senior Design Repository

This repository contains the 2025-2026 A-dec senior design software for a camera-guided dental light/gimbal prototype. The codebase includes the current integrated demo, camera and gesture experiments, gimbal SerialAPI work, audio/transcription experiments, white-balance tests, and older development scripts kept for reference.

The repository has a large amount of exploratory work. For new development, start with `camera/testing/full_demo.py` and the SerialAPI bridge in `camera/testing/SerialLibrary/serialAPI`.

## Current Primary Files

| Path | Purpose |
| --- | --- |
| `camera/testing/full_demo.py` | Main integrated demo. Combines face/mouth tracking, hand gestures, gimbal speed control, motor enable/disable gestures, patient photo capture, audio recording/transcription hooks, ADC light-mode detection, camera profiles, and QR-based white balance. |
| `camera/testing/gesture_based_speed_track.py` | Earlier gesture-based speed-tracking demo. Similar tracking and gesture ideas, but less complete than `full_demo.py`. |
| `camera/testing/SerialLibrary/serialAPI/libsimplebgc.so` | Precompiled Linux shared library loaded from Python with `ctypes`. This is the bridge between Python and the SimpleBGC C SerialAPI. |
| `camera/testing/SerialLibrary/serialAPI/simplebgc_shim.c` | Small C wrapper exposing simple functions such as `bgc_init`, `bgc_control_speeds`, and `bgc_set_motors` for Python. |
| `camera/testing/SerialLibrary/basic_librrary_compiler.txt` | Notes/commands for rebuilding `libsimplebgc.so` on the Raspberry Pi/Linux target. |
| `gesture_demo_flowchart.txt` | ASCII flowchart describing the gesture-based tracking control flow. |
| `GITGUIDE.md` | Beginner-friendly Git/GitHub Desktop workflow for the team. |

## Repository Layout

| Directory | Purpose |
| --- | --- |
| `.github/` | Repository metadata such as `CODEOWNERS`. |
| `camera/testing/` | Main Python development area for camera tracking, gestures, photos, audio, ADC tests, white balance, and the Python-facing SerialAPI library. |
| `camera/testing/adc/` | ADS1115/light-mode input tests. These scripts helped validate analog inputs used to detect dental-light mode. |
| `camera/testing/audio_recordings/` | Runtime output folder for recorded audio and optional Vosk speech-recognition model files. |
| `camera/testing/av_recordings/` | Runtime output folder for audio/video recording experiments. |
| `camera/testing/camera_profiles/` | JSON camera settings used by the demos to switch exposure, white balance, focus, and related OpenCV camera properties. |
| `camera/testing/camera_test/` | Simple camera viewing/test utilities. |
| `camera/testing/deprecated/` | Older tracking approaches retained as historical reference. Do not treat these as current unless intentionally reviving old work. |
| `camera/testing/gesture_control/` | Hand-gesture and exposure-control experiments that informed the final gesture interface. |
| `camera/testing/motor_test/` | Small motor/gimbal movement tests. |
| `camera/testing/poster_captures/` | Runtime output folder for screenshots/captures used in reports, demos, and posters. |
| `camera/testing/SerialLibrary/` | Python test scripts and the local SimpleBGC shared-library build used by the Python demos. |
| `camera/testing/speed_control/` | Development scripts for speed-mode tracking, compliance mode, dynamic stable boxes, button/pinch gestures, and related controller experiments. |
| `camera/testing/transcript/` | Audio recording and Vosk transcription experiments. |
| `camera/testing/white_balance/` | Camera exposure and white-balance calibration experiments. |
| `DemoLaunch/` | Standalone Linux C example project for controlling the gimbal with the SimpleBGC SerialAPI. Useful as a pure-C reference. |
| `gimbal/` | Vendor SimpleBGC SerialAPI source, Linux examples, gimbal profile/settings files, and build notes. |

## Main Demo Overview

`full_demo.py` is the best representation of the current system. At a high level, it:

1. Opens the face-tracking camera and center/photo camera.
2. Initializes MediaPipe FaceMesh for mouth tracking and MediaPipe Hands for gesture detection.
3. Loads the SimpleBGC shared library with `ctypes`.
4. Starts with the gimbal in a locked/hold state.
5. Detects supported hand gestures and updates the active mode.
6. Tracks either the patient mouth center or the user fingertip.
7. Converts pixel error into yaw/pitch speed commands.
8. Smooths and clamps commands before sending them to the gimbal.
9. Saves patient photos, audio recordings, transcripts, and preview captures when triggered.
10. Stops motion, turns motors off, releases cameras, and closes the display on exit.

Supported gesture behavior in the current demo:

| Gesture | Behavior |
| --- | --- |
| Pinch with index finger | Switches to fingertip tracking. |
| Four fingers | Returns to mouth tracking. |
| Closed fist | Locks/holds the gimbal position. |
| Two fingers | Starts a photo countdown and captures a patient photo. |
| Three fingers | Starts or stops audio recording. |
| Two closed fists | Turns gimbal motors off. |
| Thumbs up | Re-enables gimbal motors after shutdown. |

## SerialAPI and Python Integration

The gimbal is controlled with the BaseCam/SimpleBGC SerialAPI, which is written in C. Python cannot directly call normal C source files, so this project compiles the needed C code into a Linux shared library, then loads that library from Python using `ctypes`.

The integration has three layers:

1. Vendor SerialAPI source:
   `gimbal/serial_API_BoardExamples/serialAPI/` contains the original SimpleBGC SerialAPI source tree and examples.

2. Project C shim:
   `camera/testing/SerialLibrary/serialAPI/simplebgc_shim.c` wraps the larger SerialAPI into a small set of functions that are easy to call from Python.

3. Python loader:
   `full_demo.py` and the SerialLibrary test scripts load `libsimplebgc.so` with `ctypes.CDLL`, define the argument/return types, and call the shim functions.

### Why the C Shim Exists

The vendor SerialAPI is powerful but complex. Python only needs a few high-level actions:

| Shim Function | Purpose |
| --- | --- |
| `bgc_init()` | Initializes the SimpleBGC device and configures control behavior. |
| `bgc_control_speeds(roll_dps, pitch_dps, yaw_dps)` | Sends speed-mode commands in degrees per second. This is the main control path used by tracking. |
| `bgc_control_angles(roll_deg, pitch_deg, yaw_deg)` | Sends absolute angle commands. Kept for testing/legacy behavior. |
| `bgc_set_motors(on_off)` | Turns motors on or off. Used by safety/compliance gestures. |
| `bgc_deinit()` | Deinitializes the SerialAPI connection. |

The shim also hides C structures such as `sbgcGeneral_t`, `sbgcControl_t`, `sbgcControlConfig_t`, and `sbgcConfirm_t` so Python does not need to manage vendor-specific memory layouts.

### Why Speed Mode Is Used

The active tracking demos use `bgc_control_speeds()` rather than absolute angle commands. Speed mode is better for visual tracking because the controller only asks the gimbal to move at a certain rate toward the target. This avoids sudden startup jumps that can happen when sending absolute angle targets, especially if the controller interprets zero degrees as a physical target position.

### Important SerialAPI Files

| Path | Purpose |
| --- | --- |
| `camera/testing/SerialLibrary/serialAPI/sbgc32.c` | Aggregates the SimpleBGC SerialAPI implementation. With `SBGC_NEED_SOURCES_MAKE` enabled, this brings in the nested source files needed for the library. |
| `camera/testing/SerialLibrary/serialAPI/sbgc32.h` | Main SimpleBGC header used by the shim. |
| `camera/testing/SerialLibrary/serialAPI/serialAPI_Config.h` | Project-specific SerialAPI configuration. Enables Linux driver support, protocol version, selected modules, debug behavior, and `/dev/ttyUSB0` serial settings. |
| `camera/testing/SerialLibrary/serialAPI/simplebgc_shim.c` | Current Python-facing wrapper compiled into the shared object. |
| `camera/testing/SerialLibrary/serialAPI/simplebgc_shim2.c` | Experimental/alternate shim with additional angle readback and hold helpers. Not the primary build target unless intentionally selected. |
| `camera/testing/SerialLibrary/serialAPI/libsimplebgc.so` | Built shared object loaded by Python on the Raspberry Pi/Linux environment. |
| `camera/testing/SerialLibrary/libraryBGCtester.py` | Simple Python test for loading the shared library and sending motor commands. |
| `camera/testing/SerialLibrary/read_board_angle_tester.py` | Test utility for reading board/gimbal angle data. |
| `camera/testing/SerialLibrary/Zero_Based_Tester.py` | Test utility for zero-based movement behavior. |
| `camera/testing/SerialLibrary/fov_tester.py` | Field-of-view/gimbal behavior test utility. |

### Rebuilding the Shared Library

Run these commands on the Raspberry Pi/Linux target from:

```bash
cd ~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI
```

Remove old artifacts:

```bash
rm -f libsimplebgc.so sbgc32.o simplebgc_shim.o serialAPI_MakeCpp.o
```

Compile the SerialAPI and shim as position-independent objects:

```bash
gcc -fPIC -c sbgc32.c -o sbgc32.o
gcc -fPIC -c simplebgc_shim.c -o simplebgc_shim.o
```

Link them into a shared library:

```bash
gcc -shared -o libsimplebgc.so sbgc32.o simplebgc_shim.o -lpthread
```

The result is:

```text
camera/testing/SerialLibrary/serialAPI/libsimplebgc.so
```

Python loads that file by path and exposes it through small wrapper functions. Example pattern:

```python
import ctypes

lib = ctypes.CDLL("camera/testing/SerialLibrary/serialAPI/libsimplebgc.so")

lib.bgc_init.argtypes = []
lib.bgc_init.restype = ctypes.c_int

lib.bgc_control_speeds.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]
lib.bgc_control_speeds.restype = ctypes.c_int

lib.bgc_init()
lib.bgc_control_speeds(0.0, 5.0, -10.0)
```

### Serial Port Notes

`serialAPI_Config.h` currently configures the Linux driver to use:

```c
#define SBGC_SERIAL_PORT "/dev/ttyUSB0"
#define SBGC_SERIAL_SPEED B115200
```

On the Raspberry Pi, the connected SimpleBGC controller must appear at `/dev/ttyUSB0`, or this value must be changed and the shared library rebuilt. The serial device may also need permissions such as:

```bash
sudo chmod a+rwx /dev/ttyUSB0
```

## Camera and Vision Components

The camera side of the project is mainly organized under `camera/testing/`.

| Area | Purpose |
| --- | --- |
| Face tracking | Uses MediaPipe FaceMesh to locate mouth landmarks. The mouth center becomes the target for the dental light. |
| Hand gestures | Uses MediaPipe Hands to detect pinch, fist, two-finger, three-finger, four-finger, thumbs-up, and two-fist gestures. |
| Dynamic stable box | Adjusts the no-motion zone based on apparent face distance, estimated from eye landmark spacing. |
| Speed controller | Converts target offset from pixels to angular error using camera field-of-view, then sends proportional speed commands to the gimbal. |
| Camera profiles | JSON files store OpenCV camera settings for different lighting states. |
| White balance | QR-code-based calibration scripts and runtime helpers tune camera white balance. |
| ADC light mode | ADS1115 tests read hardware light-mode signals and select the correct camera profile. |

## Runtime Outputs and Generated Files

Several folders contain generated outputs from demos and tests:

| Path | Contents |
| --- | --- |
| `camera/testing/audio_recordings/` | Audio recordings and transcript outputs from gesture-triggered recording. |
| `camera/testing/av_recordings/` | Audio/video experiment outputs. |
| `camera/testing/poster_captures/` | Saved preview frames and report/poster images. |
| `camera/testing/face_track_log.txt` | Runtime log file written by tracking scripts. |
| `camera/testing/Board_Serial_Command_Test.txt` | Serial/motor test log file. |

These files are useful during development but should be reviewed before committing because they can become large or hardware-specific.

## Legacy and Experimental Work

This repository intentionally preserves old development work because many scripts document how the system evolved. In general:

| Area | Use |
| --- | --- |
| `camera/testing/deprecated/` | Historical tracking scripts and angle-mode experiments. Useful for reference only. |
| `camera/testing/speed_control/` | Iterations of speed-based tracking before the integrated demo. Useful when debugging controller behavior. |
| `camera/testing/gesture_control/` | Gesture and exposure experiments that led into the final demo. |
| `DemoLaunch/` | Standalone C example for proving the gimbal can be controlled without Python. |
| `gimbal/serial_API_BoardExamples/` | Original/vendor SerialAPI reference implementation and examples. |

## Development Notes

- Run the integrated system from the Raspberry Pi/Linux environment where the cameras, ADS1115, and SimpleBGC controller are connected.
- Rebuild `libsimplebgc.so` on the target architecture. A library compiled on one architecture may not load on another.
- If the Python demo cannot load the library, confirm the `LIB_PATH` in the script points to the built `.so` file.
- If motor commands fail, confirm `/dev/ttyUSB0` exists, permissions are correct, and the controller baud rate matches `B115200`.
- Keep new active work in clearly named files or folders. Move superseded experiments into `deprecated/` once they are no longer part of the current workflow.
