@echo off
rem  Mother machine pipeline - opens the graphical interface in the web browser.
rem  Keep this window open while you use it; close it to stop the interface.
setlocal
cd /d "%~dp0"
title Mother machine pipeline
set "CONDA="
for %%P in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if not defined CONDA if exist "%%~P\condabin\conda.bat" set "CONDA=%%~P\condabin\conda.bat"
)
if not defined CONDA (
    for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA set "CONDA=%%C"
)
if not defined CONDA (
    echo  conda was not found: double-click install_windows.bat first.
    pause
    exit /b 1
)
call "%CONDA%" activate mothermachine
if errorlevel 1 (
    echo  The environment "mothermachine" does not exist: double-click install_windows.bat first.
    pause
    exit /b 1
)
echo.
echo  Mother machine pipeline
echo  The interface opens in your web browser at http://localhost:8501
echo  Keep this window open while you use it; close it to stop the interface.
echo.
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start "" http://localhost:8501"
cd scripts
python -m streamlit run interfaz.py --server.headless true --server.address localhost --server.port 8501 --browser.gatherUsageStats false --client.toolbarMode viewer
pause
