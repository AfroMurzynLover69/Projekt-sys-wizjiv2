#!/usr/bin/env bash

set -u

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Linux only."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

pause_if_interactive() {
  if [[ -t 0 ]]; then
    read -r _
  fi
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return 0
  fi

  return 1
}

if ! PYTHON_CMD="$(find_python)"; then
  echo "Python not found."
  pause_if_interactive
  exit 1
fi

if ! "$PYTHON_CMD" --version >/dev/null 2>&1; then
  echo "Python error."
  pause_if_interactive
  exit 1
fi

VENV_DIR="$SCRIPT_DIR/venv"
if [[ ! -d "$VENV_DIR" ]]; then
  if ! "$PYTHON_CMD" -m venv "$VENV_DIR"; then
    echo "venv error."
    pause_if_interactive
    exit 1
  fi
fi

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "venv error."
  pause_if_interactive
  exit 1
fi

REQ_FILE="$SCRIPT_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "requirements.txt missing."
  pause_if_interactive
  exit 1
fi

NEED_INSTALL=0
if [[ ! -f "$VENV_DIR/.deps_linux_cpu_ok" ]]; then
  NEED_INSTALL=1
fi

if ! "$VENV_PY" -c "import importlib.util, sys; req=['fast_alpr','cv2','onnxruntime','ultralytics','lap','fastapi','uvicorn']; miss=[m for m in req if importlib.util.find_spec(m) is None]; sys.exit(0 if not miss else 1)" >/dev/null 2>&1; then
  NEED_INSTALL=1
fi

if [[ "$NEED_INSTALL" -eq 1 ]]; then
  if ! "$VENV_PY" -m pip install --upgrade pip; then
    echo "pip error."
    pause_if_interactive
    exit 1
  fi

  if ! "$VENV_PY" -m pip install --upgrade -r "$REQ_FILE"; then
    echo "install error."
    pause_if_interactive
    exit 1
  fi

  if ! "$VENV_PY" -c "import onnxruntime as ort; print('onnxruntime providers=', ort.get_available_providers())"; then
    echo "onnxruntime error."
    pause_if_interactive
    exit 1
  fi

  : > "$VENV_DIR/.deps_linux_cpu_ok"
fi

echo "http://127.0.0.1:8000"
"$VENV_PY" -m uvicorn app.web:app --host 127.0.0.1 --port 8000 --reload --reload-exclude "venv/*" --reload-exclude "web_data/*"
APP_STATUS=$?

pause_if_interactive
exit "$APP_STATUS"
