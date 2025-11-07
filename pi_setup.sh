# Minimal Pi 5 setup (Python 3.10.13 via pyenv, venv + required packages)

echo "==> APT update + core build deps..."
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
  build-essential cmake git curl wget unzip \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libv4l-dev v4l-utils libgtk-3-dev \
  libatlas-base-dev gfortran python3-openssl \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev libffi-dev xz-utils

# Use XCB for OpenCV windows (current shell only; no .bashrc write)
export QT_QPA_PLATFORM=xcb

# -------- pyenv (script-scoped) ----------
if [ ! -d "$HOME/.pyenv" ]; then
  echo "==> Installing pyenv..."
  curl https://pyenv.run | bash
fi
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# -------- Python 3.10.13 ----------
if ! pyenv versions --bare | grep -q '^3\.10\.13$'; then
  echo "==> Building Python 3.10.13 (limited jobs, no swap)..."
  MAKEFLAGS="-j2" CONFIGURE_OPTS="--enable-shared" pyenv install 3.10.13
fi

# Use 3.10.13 inside the project only
mkdir -p "$HOME/senior_design/camera/testing" "$HOME/senior_design/models"
cd "$HOME/senior_design"
pyenv local 3.10.13
echo "==> Using: $(python -V)"

# -------- venv (clean) ----------
if [ -d "$HOME/senior_design/venv310" ]; then
  echo "==> Removing existing venv..."
  rm -rf "$HOME/senior_design/venv310"
fi
python -m venv "$HOME/senior_design/venv310"
# shellcheck disable=SC1091
source "$HOME/senior_design/venv310/bin/activate"
echo "==> Venv Python: $(python -V)"

# -------- Packages ----------
echo "==> Installing Python packages..."
pip install --upgrade pip setuptools wheel
pip install "numpy<2" "protobuf<4" pandas pyserial
pip install opencv-python==4.8.1.78 tflite-runtime==2.14.0

# MediaPipe: try PyPI first, fall back to PiWheels (works well on Pi)
if ! pip install mediapipe; then
  echo "==> Falling back to PiWheels for MediaPipe 0.10.14..."
  pip install --index-url https://www.piwheels.org/simple mediapipe==0.10.14
fi

# Log file for your tracker
touch "$HOME/senior_design/camera/testing/face_track_log.txt"

# -------- Verify ----------
echo "==> Verifying imports..."
python - <<'PY'
import importlib, sys
mods = ["cv2","mediapipe","numpy","pandas","serial"]
ok=True
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[OK] {m}")
    except Exception as e:
        ok=False
        print(f"[FAIL] {m} -> {e}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY

echo ""
echo "=============================="
echo "Setup complete."
echo "Project dir:  ~/senior_design"
echo "Venv:         ~/senior_design/venv310"
echo "Activate with: source ~/senior_design/venv310/bin/activate"
echo "Log file:     ~/senior_design/camera/testing/face_track_log.txt"
echo "=============================="
