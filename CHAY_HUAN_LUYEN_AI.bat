@echo off
setlocal
cd /d "%~dp0"
title NTN LEO - LSTM Training

echo ================================================
echo   NTN LEO - STEP 3: HUAN LUYEN AI LSTM
echo ================================================

if not exist "data\ntn_link_quality_dataset.csv" (
  echo [LOI] Khong tim thay data\ntn_link_quality_dataset.csv
  echo Hay tao Dataset trong trang 02, tai CSV va dat vao thu muc data.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Tao moi truong Python 3.12...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Khong tao duoc .venv. Kiem tra Python 3.12 da duoc cai dat chua.
    pause
    exit /b 1
  )
)

echo [1/3] Cap nhat pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto install_error

echo [2/3] Cai thu vien AI...
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements_ai.txt
if errorlevel 1 goto install_error

echo [3/3] Bat dau huan luyen LSTM...
.venv\Scripts\python.exe ai\train_lstm.py --csv data\ntn_link_quality_dataset.csv --output-dir results\step3_lstm
if errorlevel 1 goto train_error

echo.
echo Hoan tat. Kiem tra models\ va results\step3_lstm\
pause
exit /b 0

:install_error
echo Co loi khi cai thu vien. Kiem tra ket noi Internet va Python.
pause
exit /b 1

:train_error
echo Huan luyen chua thanh cong. Doc thong bao loi o phia tren.
pause
exit /b 1
