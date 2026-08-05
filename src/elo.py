import pandas as pd
import numpy as np

def compute_elo(df, initial_rating=1500, k=30, margin_multiplier=0.5):
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
        
        r_home_before = ratings[home]
        r_away_before = ratings[away]
        
        # No home advantage
        r_home = r_home_before
        r_away = r_away_before
        
        home_elo_list.append(r_home_before)
        away_elo_list.append(r_away_before)
        
        exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        exp_away = 1 - exp_home
        
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score == away_score:
            actual_home = 0.5
            actual_away = 0.5
        else:
            actual_home = 0.0
            actual_away = 1.0
        
        # ===== TOURNAMENT IMPORTANCE WEIGHTING =====
        k_effective = k
        
        # Base multipliers
        if 'World Cup' in tournament:
            k_effective = k * 2.5
        elif 'Euro' in tournament or 'European' in tournament:
            k_effective = k * 2.0
        elif 'Copa America' in tournament:
            k_effective = k * 1.8
        elif 'Africa Cup' in tournament or 'AFCON' in tournament:
            k_effective = k * 1.6
        elif 'Asian Cup' in tournament:
            k_effective = k * 1.4
        elif 'CONCACAF Gold Cup' in tournament:
            k_effective = k * 1.3
        # Qualifiers
        elif 'qualifier' in tournament.lower():
            if 'UEFA' in tournament or 'European' in tournament:
                k_effective = k * 1.8
            elif 'CONMEBOL' in tournament:
                k_effective = k * 1.6
            elif 'CAF' in tournament or 'Africa' in tournament:
                k_effective = k * 1.3
            elif 'AFC' in tournament or 'Asian' in tournament:
                k_effective = k * 1.2
            else:
                k_effective = k * 1.1
        elif 'Friendly' in tournament:
            k_effective = k * 0.5
        # Nations League
        elif 'Nations League' in tournament:
            k_effective = k * 1.5
        else:
            k_effective = k
        
        goal_diff = abs(home_score - away_score)
        if goal_diff > 0:
            k_effective = k_effective * (1 + margin_multiplier * np.log(goal_diff + 1))
        
        ratings[home] += k_effective * (actual_home - exp_home)
        ratings[away] += k_effective * (actual_away - exp_away)
        
        home_elo_after_list.append(ratings[home])
        away_elo_after_list.append(ratings[away])
    
    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    df['home_elo_after'] = home_elo_after_list
    df['away_elo_after'] = away_elo_after_list
    
    return df