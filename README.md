# 🏆 FIFA World Cup 2026 Predictor

[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/vinay-jawahrani/fifa_wc_2026_predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A full-stack machine learning application that predicts match outcomes and simulates the FIFA World Cup 2026 tournament using Elo ratings, XGBoost, and Monte Carlo simulations.

## 🚀 Live Demo

- **Live App:** [fifa-wc-2026-prediction-model.up.railway.app](https://fifa-wc-2026-prediction-model.up.railway.app)

## 📊 Features

- **Match Prediction** – Predicts outcomes (Home Win / Draw / Away Win) using XGBoost
- **Tournament Simulation** – Simulates the entire World Cup 2026 with Monte Carlo method
- **Elo Ratings** – Calculates dynamic team strengths from historical matches (1872–2025)
- **Tournament Weighting** – UEFA Euros and World Cup qualifiers weighted higher than friendlies
- **Poisson Goal Distribution** – Realistic scoreline generation
- **Interactive Dashboard** – Built with Streamlit for easy exploration
- **Team Form & Momentum** – Uses recent match results and goal difference

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML Models** | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Deployment** | Railway |

## 📁 Project Structure
fifa_wc_2026_predictor/
├── app/
│ └── dashboard.py # Main Streamlit application
├── src/
│ ├── data_loader.py # Loads match data
│ ├── elo.py # Elo rating system
│ ├── features.py # Feature engineering
│ ├── train_model.py # XGBoost training script
│ └── world_cup_2026.py # World Cup simulation logic
├── data/
│ └── raw/
│ └── results.csv # Historical match data
├── models/
│ ├── xgboost_model.pkl
│ ├── scaler.pkl
│ └── feature_columns.pkl
├── requirements.txt
├── runtime.txt
├── railway.json
└── README.md

text

## 📦 Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vinay-jawahrani/fifa_wc_2026_predictor.git
   cd fifa_wc_2026_predictor
Create and activate a virtual environment:

bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
Install dependencies:

bash
pip install -r requirements.txt
Download the dataset:

Download results.csv from Kaggle

Place it in data/raw/results.csv

Train the model:

bash
python src/train_model.py
Run the dashboard locally:

bash
streamlit run dashboard.py
Open your browser and go to http://localhost:8501

🧪 How It Works
1. Elo Ratings
Each team starts with a base Elo rating (1500). After each match, ratings are updated based on:

Match result (win/draw/loss)

Tournament importance (World Cup: 2.5x, Euros: 2.2x, Friendlies: 0.5x)

Goal difference (margin of victory)

2. Match Prediction
The XGBoost model uses:

elo_diff – Elo rating difference

home_form – Points per game in last 5 matches

away_form – Points per game in last 5 matches

home_gd_avg – Average goal difference in last 5 matches

away_gd_avg – Average goal difference in last 5 matches

Tournament type (friendly, qualifier, etc.)

3. Tournament Simulation
The Monte Carlo simulation runs the World Cup 10,000+ times:

Group stage (12 groups of 4 teams)

Knockout rounds (Round of 32 → Final)

Third-place match

Tracks win probabilities for each team

📊 Results
The app displays:

Group Standings – Points, goal difference, goals scored/conceded

Knockout Bracket – Full tournament tree with scores and penalties

Win Probabilities – Monte Carlo simulation results

Match Predictions – Head-to-head predictions for any two teams

🚀 Deployment
This project is deployed on Railway:

Connect your GitHub repository

Add environment variable: PYTHON_VERSION=3.11.8

Set Build Command: pip install -r requirements.txt

Set Start Command: streamlit run dashboard.py --server.port 8000

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

👤 Author
Vinay Jawahrani

🌟 Show Your Support
If you found this project useful, give it a ⭐ on GitHub!
