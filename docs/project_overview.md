# NBA Analytics Project — Progress Log

This document tracks the development of my NBA analytics project.  
It outlines each step taken, the reasoning behind key decisions, and the technical details behind the work.  
I will continue updating this file as the project evolves.

---

## Table of Contents
- [1. Project Direction and Goals](#1-project-direction-and-goals)
- [2. Initial Planning and Project Architecture](#2-initial-planning-and-project-architecture)
- [3. Data Ingestion and Early Development](#3-data-ingestion-and-early-development)
- [4. Data Cleaning and Standardization](#4-data-cleaning-and-standardization)
- [5. Building the SQL Layer](#5-building-the-sql-layer)
- [6. Feature Engineering](#6-feature-engineering)
- [7. Predictive Models](#7-predictive-models)
- [8. Model Evaluation Suite](#8-model-evaluation-suite)

---

## 1. Project Direction and Goals

### 1.1 Determining the Project Direction
When I started planning this project, I wanted to build something that represented both my interests and my abilities. I needed a project that would keep me engaged long‑term, but also one that would let me demonstrate real analytics engineering skills. NBA data felt like the perfect fit. It’s something I care about, and it offers a rich, structured dataset that naturally supports the kind of work I want to showcase. Choosing this direction gives me room to explore data ingestion, modeling, analysis, and storytelling in a way that feels meaningful and technically challenging.

### 1.2 Project Goals
The goals of this project are intentionally broad so the work can evolve naturally as I learn, explore new ideas, and expand the scope over time.

#### Core Goals
- Build a project that allows me to showcase my current analytics and data engineering skills.
- Create opportunities to learn new tools, techniques, and workflows as the project grows.
- Develop a deeper understanding of how real-world data pipelines and analytics systems are designed.

#### Technical Goals
- Construct a structured database that organizes multi-season NBA data in a clean, scalable way.
- Practice working with APIs, Python, SQL, and other technologies that support modern analytics work.
- Build models, dashboards, or analytical tools that help translate raw data into meaningful insights.

#### Analytical Goals
- Explore NBA data to uncover trends, patterns, and interesting stories.
- Create visualizations and summaries that communicate insights clearly and effectively.
- Leave room for more advanced analysis—such as shot charts, player comparisons, or predictive modeling—whenever I’m ready to expand.

#### Personal and Professional Growth Goals
- Use this project as a way to continue my education and expand on the skills I’ve already built.
- Build something that reflects both my interests and my long-term career goals.
- Maintain documentation that shows how the project develops over time and how my skills progress.

---

## 2. Initial Planning and Project Architecture

### 2.1 Establishing the Project Structure
I started by setting up a simple, organized project structure that would make it easy to keep everything separated as the project grows. I didn’t want things to get messy once I started pulling in multiple datasets, so I created folders for raw data, processed data, source code, notebooks, and documentation. I also added dedicated spaces for database and modeling work since those will become major parts of the project later on.

**Example structure:**

```text
nba-analytics-project/
│
├── data/
│   ├── raw/          # data pulled directly from the API, untouched
│   └── processed/    # cleaned or organized data ready for analysis
│
├── src/
│   ├── data/         # scripts that pull in or prepare data
│   └── utils/        # small helper functions used across the project
│
├── database/         # SQL files, schema plans, and anything related to the database
│
├── models/           # analytical models or experiments I build
│
├── notebooks/        # Jupyter notebooks for exploring data and testing ideas
│
└── docs/             # project documentation, notes, and progress updates
```

- This structure gives me room to grow without having to reorganize everything later. As I add more datasets, build out the database, or start experimenting with models and dashboards, each part of the project already has a clear place to live.


### 2.2 Planning the Data Pipeline
I outlined the core datasets I wanted to ingest and how they would eventually connect inside a relational database. This included identifying:
- player-level data  
- team-level data  
- game-level data  
- potential future datasets (e.g., shot charts, play-by-play)

I also planned the order of ingestion, starting with foundational tables such as player master data and season-level statistics.

### 2.3 Choosing Tools and Technologies
I selected tools that would allow me to build a realistic analytics engineering workflow:
- **Python** for data ingestion and transformation  
- **nba_api** for accessing NBA Stats data  
- **SQL** for database modeling and analysis  
- **Markdown** for documentation  
- **VS Code** as the primary development environment  

This planning phase ensured that the project had a strong foundation before any data was collected.

---

## 3. Data Ingestion and Early Development

### 3.1 Figuring Out What Data to Pull First
I started by choosing a few core datasets that would give me a solid foundation for the rest of the project. I focused on the basics first so everything else I build later has something reliable to connect to.

Here’s what I pulled to start:

- **Player master data** – basic player information  
- **Player season stats** – one row per player per season  
- **Team information** – team IDs, names, and metadata  
- **Team game logs** – one row per team per game  


### 3.2 Writing the First Ingestion Scripts
Once I knew what I needed, I wrote simple scripts to pull the data from the NBA API. I kept the code clean and straightforward so it’s easy to maintain as the project grows.

The scripts handled a few basic steps:

- Connecting to the NBA API using the `nba_api` package  
- Requesting the data from the correct endpoints  
- Converting the API response into a pandas DataFrame  
- Saving the raw data exactly as it came in  
- Creating a processed version with cleaner column names and consistent formatting  
- Storing everything in the correct folders (`raw` and `processed`)  


### 3.3 Saving and Organizing the Data
After pulling the data, I saved it into the folder structure I set up earlier. Keeping things organized from the start helps avoid confusion later on.

- **Raw data** → saved untouched in `data/raw/`  
- **Processed data** → cleaned and saved in `data/processed/`  

This separation makes it easy to track what I’ve done and gives me a clean starting point if I ever need to reprocess something.

### 3.4 Early Challenges and Things I Had to Figure Out
A few small things came up while writing the scripts:

- Some endpoints returned slightly different fields than expected  
- I had to adjust column names to keep everything consistent  
- I made decisions about how to structure the processed files  
- I tested a few different API calls to make sure I was pulling the right data  

## 4. Data Cleaning and Standardization

### 4.1 Building a Consistent Cleaning Workflow
After the raw data was ingested and organized, the next major step was creating a consistent cleaning process across all datasets. I wanted each table to follow the same conventions so they would fit together cleanly once I moved into SQL. This meant standardizing column names, enforcing consistent data types, and making sure each dataset had the keys needed for relational modeling.

A few core rules guided the cleaning process:
- Convert all column names to lowercase with underscores  
- Ensure IDs (`player_id`, `team_id`, `game_id`) were integers  
- Convert date fields to proper datetime objects  
- Add a `season` column to every dataset  
- Remove duplicates and unnecessary fields  


### 4.2 Cleaning Each Core Dataset
I applied the cleaning workflow to all four foundational datasets:

- **Player master data** — standardized names, IDs, and metadata  
- **Player season stats** — ensured consistent numeric types and season labeling  
- **Team season stats** — aligned team identifiers and cleaned season-level metrics  
- **Team game logs** — normalized game-level data and converted dates  

Each dataset now has clean, analysis-ready fields and consistent naming conventions. This step was important because it ensures that SQL joins will behave predictably and that future feature engineering won’t require rework.

### 4.3 Creating the Processed Data Layer
After cleaning each DataFrame, I saved them into the `data/processed/` directory. This gives the project a clear separation between raw and transformed data and creates a stable layer that the SQL database can rely on.

The processed layer now includes:
- `players.csv`  
- `player_stats.csv`  
- `team_stats.csv`  
- `team_game_logs.csv`  

These files represent the first fully reproducible output of the pipeline.

## 5. Building the SQL Layer

### 5.1 Choosing SQLite for the Database
I decided to use SQLite as the database engine for this project. It’s lightweight, easy to integrate with Python, and works inside VS Code (not to mention free). Since the goal is to build a portfolio-ready analytics system, SQLite provides a simple but realistic environment for relational modeling and feature engineering.

The database lives in its own folder:


This keeps the SQL layer separate from raw and processed data and mirrors how real analytics projects organize their storage layers.

### 5.2 Writing the SQL Loading Script
To load the cleaned data into SQLite, I wrote a small Python script that reads the processed CSVs and writes them into the database as tables. The script uses simple relative paths and can be run directly from the project root.

The script loads the following tables:
- `players`
- `player_stats`
- `team_stats`
- `team_game_logs`

---

## 6. Feature Engineering

### 6.1 Goals and Design Principles

With a clean SQL layer in place, the next major phase was feature engineering — transforming raw game-log data into predictive signals for machine learning models. The guiding principle throughout this phase was **no data leakage**: every feature must be computed using only information that would have been available *before* the game being predicted. This is enforced by applying a `shift(1)` before any rolling computation, so the current game's result is never included in its own feature calculation.

I created a dedicated `src/features/` directory to separate feature engineering logic from raw data ingestion and SQL loading. Features are saved to a new `data/features/` folder so they can be reused across multiple models without re-running the full pipeline.

### 6.2 Team Game Features (`src/features/team_game_features.py`)

The core feature engineering module for team-level prediction ingests `team_game_logs.csv` and produces two output tables:

**`data/features/team_game_features.csv`** — one row per team per game, with:
- `is_home` — extracted from the matchup string (`vs.` = home, `@` = away)
- `days_rest` — days since the team's last game (capped at 14; first game of season gets 7)
- `is_back_to_back` — flag for games played on consecutive days
- `season_game_num` — how many games into the season this game falls
- `cum_win_pct` — the team's season win percentage entering this game
- `opp_pts_roll5/10/20` — rolling average of points allowed (defensive signal derived from `pts - plus_minus`)
- `sos_roll10/20` — strength of schedule: rolling average of opponent win% entering each prior game
- Rolling means over the last 5, 10, and 20 games for: `pts`, `fg_pct`, `fg3_pct`, `ft_pct`, `reb`, `ast`, `stl`, `blk`, `tov`, `pf`, `plus_minus`
- Rolling win percentage over the last 5, 10, and 20 games
- Injury proxy features (see 6.5 below): `missing_minutes`, `missing_usg_pct`, `rotation_availability`, `star_player_out`

**`data/features/game_matchup_features.csv`** — one row per game (both teams merged into a single row), with home_ and away_ prefixes on all features, plus `diff_` columns for explicit home-minus-away differentials across key stats. This is the direct input for the game outcome prediction model.

### 6.3 Player Game Features (`src/features/player_features.py`)

Player-level feature engineering runs against `player_game_logs.csv` and produces `data/features/player_game_features.csv`. For each player-game it computes:
- The same context features as the team module (home/away, rest days, back-to-back)
- Rolling means and standard deviations over 5, 10, and 20 games for all major box score stats
- Season-to-date averages using an expanding window (also shift-1 to prevent leakage)
- Season-level advanced stats joined from `player_stats_advanced.csv` (usage rate, true shooting %, net rating, PIE)
- Opponent defensive context: the opposing team's rolling points allowed and net rating, joined from `team_game_features.csv` — tells the model how strong a defense the player is facing
- Player bio context: age, height, and weight from `player_bio_stats.csv` (available from 2020-21 onward)

Volatility features (rolling standard deviation for pts, min, ast, reb) were added to help the model distinguish consistent performers from high-variance players.

### 6.5 Injury Proxy Features (`src/features/injury_proxy.py`)

One of the most impactful real-world signals for game prediction is whether a team's key players are healthy. The NBA API does not expose historical injury data, so a proxy approach was built from the player game logs. The logic: if a player who was in the regular rotation (averaging 15+ minutes over their last 5 games) has no entry in the game log for a given game, they almost certainly did not play — whether due to injury, rest, or a coach's decision.

For each team-game, the module computes four features:

`missing_minutes` measures the total expected minutes from rotation players who are absent. A value of 35 is roughly equivalent to losing a starter for the entire game. `missing_usg_pct` captures the combined usage rate of those absent players, which reflects how central they are to the team's offense — a missing player at 28% usage is a far bigger deal than one at 12%. `rotation_availability` is the fraction of normal rotation minutes that are actually available, ranging from 0 (fully depleted) to 1.0 (full strength). `star_player_out` is a binary flag for games where any absent player had a season usage rate of 25% or higher, which is the common threshold for a team's primary ball-handler.

All rolling baselines use shift(1), so the "expected" minutes are computed entirely from prior games with no leakage. The module also includes `get_todays_injury_report()`, which fetches the NBA's official pre-game injury report for live predictions, and `apply_live_injuries()`, which patches a matchup feature row with confirmed scratches before feeding it to the model.

The injury features flow into `game_matchup_features.csv` as `home_missing_minutes`, `away_missing_minutes`, and a `diff_missing_minutes` differential — giving the model a direct signal of the health gap between the two teams entering a game.

### 6.4 Era Labels (`src/features/era_labels.py`)

Because the project spans 80 seasons of NBA data, treating all seasons as a single homogeneous dataset would obscure fundamental differences in how the game was played. Rule changes — the shot clock, the 3-point line, hand-check rules — didn't just shift individual stats; they changed the structural relationships between them. A model trained across all eras without any era signal could learn misleading patterns from comparing, say, 1960s pace statistics against 2020s pace statistics as if they were equivalent.

To address this, a standalone era labeling utility was built in `src/features/era_labels.py`. It maps any `season` value to one of six historically grounded eras, each anchored to a concrete rule change or stylistic inflection point:

| Era | Name | Seasons | Rule Anchor |
|-----|------|---------|-------------|
| 1 | Foundational | 1946–47 → 1953–54 | Pre-shot clock |
| 2 | Shot Clock Era | 1954–55 → 1978–79 | 24-second shot clock introduced |
| 3 | 3-Point Introduction | 1979–80 → 1993–94 | 3-point line adopted |
| 4 | Physical / Isolation | 1994–95 → 2003–04 | Hand-checking permitted |
| 5 | Open Court | 2004–05 → 2014–15 | No hand-check / defensive 3-second rules |
| 6 | 3-Point Revolution | 2015–16 → present | Analytics era, record 3PA |

The module exposes two public functions. `label_eras(df)` takes any DataFrame with a `season` column and inserts four new columns — `era_num`, `era_name`, `era_start`, `era_end` — directly after it. The `era_num` integer is particularly useful for models as it allows era to be treated as an ordinal feature or used to filter training data to a specific period. `get_era(season)` provides a single-season lookup for quick inspection.

Era labels are automatically applied in both `team_game_features.py` and `player_features.py`, so every downstream feature table and model input already carries era context without any extra steps.

---

## 7. Predictive Models

### 7.1 Model Architecture Overview

Three predictive models were built, each serving a different analytical question. All models use scikit-learn pipelines with an imputer step so they handle missing values gracefully for early-season games with limited rolling history. All are trained on a time-based train/test split (most recent two seasons held out) to simulate real-world deployment.

### 7.2 Game Outcome Prediction (`src/models/game_outcome_model.py`)

**Question:** Given two teams' recent form, who wins?

This is a binary classification problem (home win = 1, home loss = 0). The model uses `game_matchup_features.csv` as its input and trains two classifiers for comparison:
- **Logistic Regression** — fast baseline with good interpretability
- **Gradient Boosting Classifier** — 200 estimators, depth-4 trees, learning rate 0.05

The gradient boosting model consistently outperforms the logistic regression baseline. Feature importances reveal that the most predictive signals are rolling win percentage, rolling net rating (via plus_minus), home court advantage, and cumulative season win percentage. The model can also be invoked at prediction time via `predict_game(home_abbr, away_abbr)` to get win probabilities for any matchup.

### 7.3 Player Performance Prediction (`src/models/player_performance_model.py`)

**Question:** How many points / rebounds / assists will a player record in their next game?

This is a multi-target regression problem. A separate Gradient Boosting Regressor is trained per stat target (pts, reb, ast), again compared to a Ridge regression baseline. Models are filtered to players with at least 20 games in the training set to avoid overfitting on small samples. Evaluation metrics are MAE (mean absolute error) and RMSE.

The `predict_player_next_game(player_name)` helper uses each player's most recent rolling feature snapshot to generate next-game projections.

### 7.4 Playoff Odds Simulation (`src/models/playoff_odds_model.py`)

**Question:** What are each team's odds of making the playoffs, winning their conference, or winning the title?

This model uses a Monte Carlo approach — simulating the remainder of the season 10,000 times. For each remaining game, win probabilities are estimated using a Bradley-Terry strength model derived from each team's current win percentage, with a home-court advantage factor built in. After simulating the regular season, a simplified 8-team playoff bracket is run per conference to estimate conference title and championship probabilities.

The model can be run at any point in the season. Results are saved to `data/features/playoff_odds.csv`.

---

## 8. Model Evaluation Suite

### 8.1 The Case for Rigorous Evaluation

With three predictive models built, the natural next step was to evaluate them honestly. A single train/test split only tells part of the story — it shows how the model performs on two specific seasons, but it doesn't reveal whether performance is consistent over time, whether the predicted probabilities are trustworthy, or which features are actually doing the work. Three dedicated evaluation tools were built to answer these questions, all outputting to a `reports/` directory and runnable individually or together via `src/models/run_evaluation.py`.

### 8.2 Walk-Forward Backtesting (`src/models/backtesting.py`)

The walk-forward backtest is the most rigorous evaluation tool in the suite. Rather than splitting the data once, it rolls forward one season at a time: train on all seasons up to season N, test on season N+1, then advance the window and repeat. This simulates exactly how the models would have performed in real deployment — the model never sees future data, and it retrains after each season just as it would in production.

For the game outcome model, the backtest records accuracy, ROC-AUC, and Brier score for every season after a minimum training window of five seasons. This produces a full performance curve across the history of the data, making it easy to spot whether accuracy is trending upward, downward, or is stable. It also breaks results down by historical era using the era labels built in the feature engineering phase, which reveals whether the model handles different eras of basketball differently.

For the player performance models, the same rolling approach is applied to each stat target (pts, reb, ast), tracking MAE and RMSE per season alongside a naive baseline (predicting the training mean for every game). The gap between model MAE and baseline MAE is a clean measure of how much value the rolling features actually add.

All results are saved to `reports/backtest_game_outcome.csv`, `reports/backtest_player_{stat}.csv`, and a human-readable `reports/backtest_summary.txt`.

### 8.3 Calibration Analysis (`src/models/calibration.py`)

Calibration answers a question that accuracy alone cannot: when the game outcome model predicts a 70% win probability, does the home team actually win about 70% of those games? A well-calibrated model is one whose probability outputs can be trusted directly — not just for ranking predictions, but for decision-making.

Three calibration diagnostics are computed:

The **reliability diagram** (also called a calibration curve) bins predictions into deciles and plots the mean predicted probability in each bin against the actual win rate. A perfectly calibrated model lies on the diagonal; curves above it indicate the model is underconfident, and curves below indicate overconfidence.

The **Brier score** is the mean squared error between predicted probabilities and actual outcomes. It ranges from 0 (perfect) to 0.25 (completely uninformative coin flip). Because NBA games have genuine randomness, the theoretical floor for this sport is roughly 0.22–0.24 even for an optimal model.

The **Expected Calibration Error (ECE)** is a single number summarizing average miscalibration across all bins — the weighted mean absolute gap between predicted probability and actual win rate.

Beyond the raw model, the analysis also fits an isotonic regression calibration wrapper to quantify how much calibration can be improved by post-processing. Per-season Brier score trends are plotted to show whether calibration is stable over time or degrading in recent seasons. All outputs are saved to `reports/calibration/`.

### 8.4 SHAP Explainability (`src/models/model_explainability.py`)

Raw gradient boosting feature importances only tell you which features were used most often across all tree splits. They don't tell you the direction of each feature's effect or how it contributed to any specific prediction. SHAP (SHapley Additive exPlanations) fills both of those gaps.

For each model, SHAP values are computed on a sample of the test set. The global summary (beeswarm) plot shows both the importance and direction of every feature: a feature cluster shifted to the right means high values of that feature push the prediction upward; shifted left means the opposite. A diverging bar chart of mean SHAP values makes the directional story even clearer — for example, showing that high rolling win percentage strongly increases predicted home win probability while high turnover rate decreases it.

For individual game prediction explanations, the `explain_prediction(home_abbr, away_abbr)` function produces a waterfall chart showing exactly which features are pushing the model's probability estimate up or down for that specific matchup. This is the kind of output that makes a prediction feel explainable rather than like a black box.

Because the SHAP library is an optional dependency (`pip install shap`), the module includes a graceful fallback to permutation importance from scikit-learn, which produces comparable directional charts without the extra install.

All charts and feature direction tables are saved to `reports/explainability/`.
