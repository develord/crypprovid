@echo off
echo ===============================================
echo  Starting Crypto Predictions API Server
echo ===============================================
echo.
echo Server will be available at:
echo   - API: http://localhost:8000
echo   - Swagger Docs: http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo.
echo Press CTRL+C to stop the server
echo ===============================================
echo.

cd /d "%~dp0"
python main.py

pause
