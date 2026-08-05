import pandas as pd
import numpy as np

def compute_elo(df: pd.DataFrame, 
                initial_rating: float = 1500.0, 
                k: float = 30.0,
                home_advantage: float = 100.0,
                margin_multiplier: float = 0.5,
                friendly_reduction: float = 0.5) -> pd.DataFrame:
    """
    Compute Elo ratings for each team over time.
    
    Returns:
        df with columns: 'home_elo', 'away_elo', 'home_elo_after', 'away_elo_after'
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    ratings = {}
    home_elo_list = []
    away_elo_list = []
    home_elo_after_list = []
    away_elo_after_list = []
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        tournament = row.get('tournament', '')
        
        if home not in ratings:
            ratings[home] = initial_rating
        if away not in ratings:
            ratings[away] = initial_rating
        
        # Get current ratings (before match)
        r_home_before = ratings[home]
        r_away_before = ratings[away]
        
        # Apply home advantage
        r_home = r_home_before + home_advantage
        r_away = r_away_before
        
        # Store pre-match Elo
        home_elo_list.append(r_home_before)
        away_elo_list.append(r_away_before)
        
        # Expected scores
        exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        exp_away = 1 - exp_home
        
        # Actual result
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score == away_score:
            actual_home = 0.5
            actual_away = 0.5
        else:
            actual_home = 0.0
            actual_away = 1.0
        
        # Adjust K-factor
        k_effective = k
        if 'Friendly' in tournament:
            k_effective = k * friendly_reduction
        
        goal_diff = abs(home_score - away_score)
        if goal_diff > 0:
            k_effective = k_effective * (1 + margin_multiplier * np.log(goal_diff + 1))
        
        # Update ratings
        ratings[home] += k_effective * (actual_home - exp_home)
        ratings[away] += k_effective * (actual_away - exp_away)
        
        # Store post-match Elo
        home_elo_after_list.append(ratings[home])
        away_elo_after_list.append(ratings[away])
    
    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    df['home_elo_after'] = home_elo_after_list
    df['away_elo_after'] = away_elo_after_list
    
    return df