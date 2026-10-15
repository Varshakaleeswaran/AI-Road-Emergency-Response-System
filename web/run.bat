@echo off
echo ResQLink AI — Web Portal
echo ========================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting server at http://127.0.0.1:5000
echo Default login: admin / Admin@ResQLink1
echo.
set RESQLINK_DEMO_MODE=1
python flask_app.py
pause
