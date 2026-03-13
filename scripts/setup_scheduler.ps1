<#
.SYNOPSIS
    Sets up Windows Task Scheduler tasks for the NBA analytics pipeline.
.DESCRIPTION
    Creates three scheduled tasks:
    1. NBA-Pipeline-Full (4:00 AM) - Full pipeline after games complete
    2. NBA-Pipeline-Injuries (11:30 AM) - Midday injuries + odds refresh
    3. NBA-Pipeline-Pretip (6:30 PM) - Pre-tip final predictions
.NOTES
    Run as Administrator: .\scripts\setup_scheduler.ps1
    To remove: .\scripts\setup_scheduler.ps1 -Remove
#>

param(
    [switch]$Remove,
    [string]$ProjectDir = "C:\Users\Spencer\OneDrive\Desktop\GIT\nba-analytics-project"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$runner = Join-Path $ProjectDir "scripts\pipeline_runner.py"

# Verify paths exist
if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python"
    exit 1
}
if (-not (Test-Path $runner)) {
    Write-Error "Pipeline runner not found at $runner"
    exit 1
}

$tasks = @(
    @{
        Name = "NBA-Pipeline-Full"
        Time = "4:00AM"
        Mode = "full"
        Description = "Full NBA pipeline: fetch data, build features, predictions, deploy dashboard"
    },
    @{
        Name = "NBA-Pipeline-Injuries"
        Time = "11:30AM"
        Mode = "injuries_odds"
        Description = "Midday refresh: injuries + odds update"
    },
    @{
        Name = "NBA-Pipeline-Pretip"
        Time = "6:30PM"
        Mode = "pretip"
        Description = "Pre-tip: final odds, predictions, picks, deploy"
    }
)

if ($Remove) {
    foreach ($task in $tasks) {
        if (Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false
            Write-Host "Removed: $($task.Name)" -ForegroundColor Yellow
        } else {
            Write-Host "Not found: $($task.Name)" -ForegroundColor Gray
        }
    }
    Write-Host "`nAll NBA pipeline tasks removed." -ForegroundColor Green
    exit 0
}

foreach ($task in $tasks) {
    # Remove existing task if present
    if (Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false
        Write-Host "Replacing existing task: $($task.Name)" -ForegroundColor Yellow
    }

    $action = New-ScheduledTaskAction `
        -Execute $python `
        -Argument "$runner --mode $($task.Mode)" `
        -WorkingDirectory $ProjectDir

    $trigger = New-ScheduledTaskTrigger -Daily -At $task.Time

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1)

    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description $task.Description `
        -RunLevel Highest | Out-Null

    Write-Host "Created: $($task.Name) at $($task.Time) ($($task.Mode))" -ForegroundColor Green
}

Write-Host "`nAll NBA pipeline tasks configured." -ForegroundColor Green
Write-Host "View in Task Scheduler or run: Get-ScheduledTask -TaskName 'NBA-Pipeline-*'" -ForegroundColor Cyan
