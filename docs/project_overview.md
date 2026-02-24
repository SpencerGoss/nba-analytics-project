# NBA Analytics Project — Progress Log

This document tracks the development of my NBA analytics project.  
It outlines each step taken, the reasoning behind key decisions, and the technical details behind the work.  
I will continue updating this file as the project evolves.

---

## Table of Contents
- [1. Project Direction and Goals](#1-project-direction-and-goals)
- [2. Initial Planning and Project Architecture](#2-initial-planning-and-project-architecture)
- [3. Data Ingestion and Early Development](#3-data-ingestion-and-early-development)

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

These datasets are used everywhere in NBA analytics, so getting them in place early made the most sense.

### 3.2 Writing the First Ingestion Scripts
Once I knew what I needed, I wrote simple scripts to pull the data from the NBA API. I kept the code clean and straightforward so it’s easy to maintain as the project grows.

The scripts handled a few basic steps:

- Connecting to the NBA API using the `nba_api` package  
- Requesting the data from the correct endpoints  
- Converting the API response into a pandas DataFrame  
- Saving the raw data exactly as it came in  
- Creating a processed version with cleaner column names and consistent formatting  
- Storing everything in the correct folders (`raw` and `processed`)  

The goal at this stage was just to get the data flowing into the project in a clean, repeatable way.

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

All of this helped me get the foundation right before moving on to the next steps.



