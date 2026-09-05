@echo off
echo Starting LocalGPT Frontend on http://localhost:3000 ...
cd /d "%~dp0frontend"
set NEXT_TELEMETRY_DISABLED=1
npm run dev
pause
