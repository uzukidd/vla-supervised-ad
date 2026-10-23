@echo off
setlocal
for %%I in ("%~dp0..\..\..") do set "PROJECT_DIR=%%~fI"
wsl.exe --cd "%PROJECT_DIR%" bash tasks/nuscenes_sam3/scripts/run_bev_map.sh %*
exit /b %ERRORLEVEL%
