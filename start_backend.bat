@echo off
echo Starting LocalGPT Backend on http://localhost:8000 ...
cd /d "%~dp0backend"
"..\LLM-XRay\venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
