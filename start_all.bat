@echo off
echo Starting LocalGPT Full-Stack Platform...
start "LocalGPT Backend" cmd /k "cd /d \"%~dp0backend\" && ..\LLM-XRay\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 /nobreak >nul
start "LocalGPT Frontend" cmd /k "cd /d \"%~dp0frontend\" && npm run dev"
echo.
echo ===================================================
echo LocalGPT Services Started!
echo Frontend:   http://localhost:3000
echo Backend:    http://127.0.0.1:8000
echo Swagger UI: http://127.0.0.1:8000/docs
echo ===================================================
