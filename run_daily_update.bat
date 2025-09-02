@echo off
echo Daily Paper Update System
echo ========================

echo Installing dependencies...
py -m pip install -r requirements.txt

echo.
echo Running daily paper fetcher...
py daily_paper_fetcher.py

echo.
echo Press any key to exit...
pause > nul
