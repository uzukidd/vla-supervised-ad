@echo off
setlocal
rem Keep inference in the existing WSL vla-supervised environment.
for %%I in ("%~dp0..\..\..") do set "PROJECT_DIR=%%~fI"
wsl.exe --cd "%PROJECT_DIR%" bash tasks/nuscenes_sam3/scripts/run_scenes.sh %*
exit /b %ERRORLEVEL%
