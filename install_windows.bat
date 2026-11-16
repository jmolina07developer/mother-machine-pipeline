@echo off
rem ============================================================================================
rem  Mother machine pipeline - installation for Windows (double-click this file once).
rem  1) installs Miniforge (conda) in your user folder if it is not there yet;
rem  2) creates the environment "mothermachine" with all the packages (~10 min, needs internet);
rem  3) puts a "Mother machine" shortcut on the Desktop.
rem  Running it again updates the packages.
rem ============================================================================================
setlocal
cd /d "%~dp0"
title Mother machine pipeline - installation
echo.
echo  Mother machine pipeline - installation
echo  --------------------------------------
echo.
call :buscar_conda
if not defined CONDA (
    echo  conda ^(Miniforge^) is not installed on this computer.
    choice /c YN /m " Download and install Miniforge now (about 100 MB, in your user folder)"
    if errorlevel 2 goto :sin_conda
    echo.
    echo  Downloading Miniforge...
    curl -L --fail -o "%TEMP%\Miniforge3-Windows-x86_64.exe" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
    if errorlevel 1 goto :error_descarga
    echo  Installing Miniforge in %USERPROFILE%\miniforge3 ^(1-2 min^)...
    start /wait "" "%TEMP%\Miniforge3-Windows-x86_64.exe" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%USERPROFILE%\miniforge3
    call :buscar_conda
    if not defined CONDA goto :sin_conda
)
echo  conda: %CONDA%
echo.
call "%CONDA%" env list | findstr /b /c:"mothermachine " >nul
if errorlevel 1 (
    echo  Creating the environment "mothermachine" ^(about 10 minutes^)...
    call "%CONDA%" env create -f environment.yml
) else (
    echo  Updating the environment "mothermachine"...
    call "%CONDA%" env update -f environment.yml --prune
)
if errorlevel 1 goto :error_entorno
echo.
echo  Checking the installation...
call "%CONDA%" run -n mothermachine python scripts\check_install.py
echo.
echo  Creating the shortcut "Mother machine" on the Desktop...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$w = New-Object -ComObject WScript.Shell; $s = $w.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Mother machine.lnk'); $s.TargetPath = '%~dp0start_windows.bat'; $s.WorkingDirectory = '%~dp0'; $s.Save()"
echo.
echo  ==========================================================================
echo   Installed. To use the pipeline, double-click "Mother machine" on the
echo   Desktop (or start_windows.bat in this folder).
echo  ==========================================================================
echo.
pause
exit /b 0

:buscar_conda
set "CONDA="
for %%P in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if not defined CONDA if exist "%%~P\condabin\conda.bat" set "CONDA=%%~P\condabin\conda.bat"
)
if not defined CONDA (
    for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA set "CONDA=%%C"
)
exit /b 0

:sin_conda
echo.
echo  conda was not found. Install Miniforge yourself from
echo  https://github.com/conda-forge/miniforge  ^(Miniforge3-Windows-x86_64.exe, "Just me"^)
echo  and double-click this file again.
pause
exit /b 1

:error_descarga
echo.
echo  The download failed. Check the internet connection, or install Miniforge yourself from
echo  https://github.com/conda-forge/miniforge and double-click this file again.
pause
exit /b 1

:error_entorno
echo.
echo  Creating the environment failed (see the messages above). Check the internet connection and
echo  try again; if it keeps failing, see INSTALL.md (Troubleshooting).
pause
exit /b 1
