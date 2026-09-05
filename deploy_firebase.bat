@echo off
echo ========================================================
echo Deploying ChatGPT Pro to Firebase Hosting...
echo ========================================================
echo.

cd /d "%~dp0"

echo Step 1: Checking Firebase CLI...
where firebase >nul 2>nul
if %errorlevel% neq 0 (
    echo [Notice] Firebase CLI not detected globally.
    echo Installing firebase-tools...
    npm install -g firebase-tools
)

echo.
echo Step 2: Building Frontend Application...
cd frontend
call npm run build
if %errorlevel% neq 0 (
    echo [Error] Frontend build failed.
    pause
    exit /b %errorlevel%
)
cd ..

echo.
echo Step 3: Deploying to Firebase Hosting...
firebase deploy --only hosting

echo.
echo ========================================================
echo Deployment finished! Visit your live Firebase URL.
echo ========================================================
pause
