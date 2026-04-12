@echo off
setlocal

cd /d "%~dp0"

echo [1/4] Sprawdzam Python...
python --version >nul 2>&1
if errorlevel 1 (
  echo Python nie jest dostepny w PATH.
  echo Zainstaluj Python 3.10+ i sprobuj ponownie.
  pause
  exit /b 1
)

echo [2/4] Tworze virtual environment (venv)...
if not exist venv (
  python -m venv venv
  if errorlevel 1 (
    echo Nie udalo sie utworzyc venv.
    pause
    exit /b 1
  )
)

set "VENV_PY=%CD%\venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo Nie znaleziono interpretera venv: %VENV_PY%
  pause
  exit /b 1
)
set "REQ_FILE=%CD%\requirements.txt"
if not exist "%REQ_FILE%" (
  echo Brak pliku requirements: %REQ_FILE%
  pause
  exit /b 1
)

echo [3/4] Sprawdzam venv i instaluje biblioteki...
set NEED_INSTALL=0
if not exist venv\.deps_ok set NEED_INSTALL=1

"%VENV_PY%" -c "import importlib.util, sys; req=['fast_alpr','cv2','onnxruntime','ultralytics','lap']; miss=[m for m in req if importlib.util.find_spec(m) is None]; sys.exit(0 if not miss else 1)" >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1

if "%NEED_INSTALL%"=="0" goto SKIP_INSTALL

echo Wykryto brakujace biblioteki ONNX/YOLO. Instalacja...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install --upgrade -r "%REQ_FILE%"

"%VENV_PY%" -c "import onnxruntime as ort; print('onnxruntime providers=', ort.get_available_providers())"

type nul > venv\.deps_ok
goto RUN_APP

:SKIP_INSTALL
echo Biblioteki juz sa gotowe. Pomijam instalacje.

:RUN_APP
echo [4/4] Uruchamiam aplikacje...
"%VENV_PY%" start.py

pause
exit /b 0
