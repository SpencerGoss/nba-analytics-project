# NBA Analytics Project

## Overview
This project is a full end‑to‑end NBA analytics system built to showcase my data engineering, analytics, and programming skills. It combines my interest in basketball with a structured, professional workflow that mirrors real analytics engineering practices. The project includes data ingestion, cleaning, database modeling, and the foundation for deeper analytical work.

## Why This Project?
I’ve always been an avid NBA fan and have been fascinated by the advanced analytics discussed during broadcasts. Choosing a topic I genuinely care about keeps me motivated and makes the technical work more meaningful. NBA data provides a rich, structured environment for practicing real‑world analytics skills, including data ingestion, transformation, relational modeling, and exploratory analysis.

## Project Goals
- **Showcase Technical Skills** — Demonstrate proficiency in Python, SQL, data cleaning, and analytics workflows.  
- **Learn by Doing** — Build practical experience with real-world datasets and reproducible pipelines.  
- **Document Progress** — Maintain clear, ongoing documentation of decisions, methods, and project evolution.  
- **Generate Insights** — Explore NBA data to uncover meaningful trends, patterns, and stories.  

---

## Where Things Live
- **Raw data** → `data/raw/`  
  - Contains subfolders for each dataset (`players/`, `player_stats/`, `team_stats/`, `team_game_logs/`)

- **Processed data** → `data/processed/`  
  - Cleaned CSVs used for analysis and SQL loading  
  - (`players.csv`, `player_stats.csv`, `team_stats.csv`, `team_game_logs.csv`)

- **Database** → `database/nba.db`  
  - SQLite database containing all cleaned tables

- **Data ingestion scripts** → `src/data/`  
  - (`get_player_master.py`, `get_player_stats.py`, `get_team_stats.py`, `get_game_log.py`)

- **Processing & loading scripts** → `src/processing/`  
  - (`preprocessing_data.ipynb`, `load_to_sql.py`)

- **Documentation** → `docs/`  
  - Full project description and progress log (`project_overview.md`)




