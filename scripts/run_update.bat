@echo off
REM NBA Analytics Daily Update
REM ==========================
REM Run this script directly or point Windows Task Scheduler at it.
REM Output is appended to logs\update.log so you can check what happened.
REM
REM TASK SCHEDULER SETUP (one-time, ~2 minutes):
REM   1. Open Task Scheduler (search "Task Scheduler" in Start Menu)
REM   2. Click "Create Basic Task" in the right panel
REM   3. Name: NBA Analytics Daily Update
REM   4. Trigger: Daily
REM   5. Start time: 7:00 AM  (NBA stats post overnight, usually by 3-4 AM)
REM   6. Action: Start a program
REM   7. Program: C:\Users\spenc\OneDrive\Desktop\GIT\nba-analytics-project\scripts\run_update.bat
REM   8. Finish
REM
REM To check if it ran: open logs\update.log

cd /d "C:\Users\spenc\OneDrive\Desktop\GIT\nba-analytics-project"

if not exist "logs" mkdir logs

echo. >> logs\update.log
echo ============================================================ >> logs\update.log
echo [%date% %time%] Starting NBA data update >> logs\update.log
echo ============================================================ >> logs\update.log

python update.py >> logs\update.log 2>&1

if %errorlevel% == 0 (
    echo [%date% %time%] Update finished successfully >> logs\update.log
) else (
    echo [%date% %time%] Update FAILED with error code %errorlevel% >> logs\update.log
)
