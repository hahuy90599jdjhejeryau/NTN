@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title NTN LEO Research Lab

echo ======================================================
echo       NTN LEO RESEARCH LAB - STEP 2 PRO FINAL
echo ======================================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Chua co moi truong Python. Dang tao .venv...
    where py >nul 2>nul
    if errorlevel 1 (
        echo Khong tim thay Python Launcher 'py'. Hay cai Python 3.12 va chon Add Python to PATH.
        pause
        exit /b 1
    ) else (
        py -3.12 -m venv .venv
        if errorlevel 1 py -3 -m venv .venv
        if errorlevel 1 (
            echo Khong tao duoc moi truong ao. Kiem tra Python da cai dat.
            pause
            exit /b 1
        )
    )
) else (
    echo [1/3] Da tim thay moi truong .venv.
)

echo [2/3] Kiem tra va cai cac thu vien can thiet...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/3] Dang khoi dong web NTN LEO...
echo Khi Streamlit hien dia chi Local URL, trinh duyet se mo tai http://localhost:8501
start "" http://localhost:8501
.venv\Scripts\python.exe -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Co loi khi cai dat hoac chay ung dung. Hay chup man hinh loi trong cua so nay.
pause
exit /b 1
