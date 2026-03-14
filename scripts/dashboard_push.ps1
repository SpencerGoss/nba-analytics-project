# Auto-push dashboard data to GitHub Pages after pipeline runs.
# Called by pipeline tasks as a post-step.

$ProjectDir = "C:\Users\Spencer\OneDrive\Desktop\GIT\nba-analytics-project"
$LogDir = "$ProjectDir\logs"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Set-Location $ProjectDir

# Stage dashboard data
git add dashboard/data/

# Check if there are changes to commit
$diff = git diff --cached --quiet 2>&1
if ($LASTEXITCODE -eq 0) {
    Add-Content -Path "$LogDir\deploy.log" -Value "[$timestamp] No dashboard changes to push"
    exit 0
}

$date = Get-Date -Format "yyyy-MM-dd"
git commit -m "data: daily dashboard update $date"

git push 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Add-Content -Path "$LogDir\deploy.log" -Value "[$timestamp] Dashboard data pushed to GitHub"
} else {
    Add-Content -Path "$LogDir\deploy.log" -Value "[$timestamp] ERROR: git push failed"
}
