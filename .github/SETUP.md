# GitHub Actions Setup

## 1. Add the BALLDONTLIE_API_KEY secret

The daily pipeline requires `BALLDONTLIE_API_KEY`. Add it as a repository secret:

1. Go to your repository on GitHub
2. Click **Settings** -> **Secrets and variables** -> **Actions**
3. Click **New repository secret**
4. Name: `BALLDONTLIE_API_KEY`
5. Value: your key from BallDontLie
6. Click **Add secret**

The Pinnacle odds API is keyless (guest API) — no secret needed for it.

## 2. Verify the workflow ran

- Go to **Actions** tab on GitHub
- Click **Daily Dashboard Update**
- Green checkmark = success; click any run to see step-by-step logs
- The workflow runs daily at 9:00 AM EST (14:00 UTC); you can also trigger it manually via **Run workflow**

## 3. Set up on a new PC

```bash
# Clone the repo
git clone https://github.com/<your-username>/nba-analytics-project.git
cd nba-analytics-project

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
# OR: .venv\Scripts\Activate.ps1  (PowerShell)

pip install -r requirements.txt

# Copy .env.example and fill in your API key
cp .env.example .env
# Edit .env and set BALLDONTLIE_API_KEY=your_key_here

# Run the pipeline manually
python update.py

# Serve the dashboard locally
python -m http.server 8080 --directory dashboard
```

For automated daily updates, the GitHub Actions workflow (`daily_deploy.yml`) handles
everything in the cloud — no local cron job needed.
