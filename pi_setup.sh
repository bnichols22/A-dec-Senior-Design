#!/usr/bin/env bash
# ================================================================
# Pi 5 Face Tracking Setup (Python 3.10 via pyenv + venv)
# - Installs build deps
# - Installs pyenv + Python 3.10.13
# - Creates venv and installs mediapipe, opencv, tflite-runtime, etc.
# - Sets QT to XCB to avoid cv2.imshow Wayland issues
# ================================================================
set -euo pipefail

echo "==> Updating system..."
sudo apt update
sudo apt full-upgrade -y

echo "==> Installing core build & media/GUI deps..."
sudo apt install -y \
  build-essential cmake pkg-config git curl wget unzip make \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libv4l-dev v4l-utils \
  libgtk-3-dev libcanberra-gtk* libqt5gui5 libqt5test5 libxcb-xinerama0 \
  libatlas-base-dev liblapack-dev gfortran \
  python3-openssl libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev libffi-dev liblzma-dev tk-dev llvm xz-utils

# Ensure OpenCV windows work under X11 on Raspberry Pi OS Bookworm
if ! grep -q 'QT_QPA_PLATFORM=xcb' "${HOME}/.bashrc"; then
  echo "export QT_QPA_PLATFORM=xcb" >> "${HOME}/.bashrc"
fi
export QT_QPA_PLATFORM=xcb

# ------------------------------------------------
# Install pyenv (if not already) and Python 3.10.13
# ------------------------------------------------
if [ ! -d "${HOME}/.pyenv" ]; then
  echo "==> Installing pyenv..."
  curl https://pyenv.run | bash
fi

# Add pyenv to PATH for current session and future logins
if ! grep -q 'PYENV_ROOT="$HOME/.pyenv"' "${HOME}/.bashrc"; then
  {
    echo 'export PYENV_ROOT="$HOME/.pyenv"'
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
    echo 'eval "$(pyenv init -)"'
  } >> "${HOME}/.bashrc"
fi

# Initialize pyenv for this script run
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install and select Python 3.10.13
if ! pyenv versions --bare | grep -q '^3\.10\.13$'; then
  echo "==> Building Python 3.10.13 with pyenv (this can take a while)..."
  CFLAGS="-O2" pyenv install 3.10.13
fi
pyenv global 3.10.13

# ------------------------------------------------
# Project structure + venv
# ------------------------------------------------
echo "==> Creating project structure..."
mkdir -p "${HOME}/senior_design/camera/testing" "${HOME}/senior_design/models"
cd "${HOME}/senior_design"

# Use pyenv's 3.10.13 python for the venv
if [ ! -d "${HOME}/senior_design/venv310" ]; then
  echo "==> Creating Python 3.10 venv at ~/senior_design/venv310"
  python -m venv "${HOME}/senior_design/venv310"
fi

echo "==> Activating venv..."
# shellcheck disable=SC1091
source "${HOME}/senior_design/venv310/bin/activate"

echo "==> Upgrading pip toolchain..."
pip install --upgrade pip setuptools wheel

echo "==> Installing Python dependencies..."
# Pin numpy<2 and protobuf<4 for compatibility with various wheels
pip install "numpy<2" "protobuf<4" pandas pyserial

# OpenCV (wheel), TFLite runtime, and MediaPipe on Python 3.10
# The pinned OpenCV version below is widely compatible on ARM64; adjust if needed.
pip install opencv-python==4.8.1.78 tflite-runtime==2.14.0

# MediaPipe installs cleanly on Python 3.10; if your PyPI wheel fails, try piwheels line below.
if ! pip install mediapipe; then
  echo "==> Falling back to piwheels for mediapipe..."
  pip install --index-url https://www.piwheels.org/simple mediapipe==0.10.14
fi

# Create an empty log file used by your scripts
touch "${HOME}/senior_design/camera/testing/face_track_log.txt"

echo "==> Verifying key packages..."
python - <<'PY'
import sys
mods = ["cv2","mediapipe","numpy","pandas","serial"]
ok = True
for m in mods:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {m}: {e}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY

echo ""
echo "================================================"
echo " Setup complete!"
echo " • VENV: source ~/senior_design/venv310/bin/activate"
echo " • Put your tracker under: ~/senior_design/camera/testing/"
echo " • Log file: ~/senior_design/camera/testing/face_track_log.txt"
echo "================================================"
