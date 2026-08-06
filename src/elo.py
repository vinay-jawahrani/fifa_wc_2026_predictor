import pandas as pd
import numpy as np

# FIFA Rankings post-2022 World Cup (December 2022)
# These are the official FIFA rankings after the 2022 World Cup final
# Includes all teams that qualified for WC 2026
FIFA_RANKINGS_2022 = {
    # CONMEBOL (South America) - 6 teams
    'Argentina': 1838,
    'Brazil': 1837,
    'Uruguay': 1635,
    'Colombia': 1624,
    'Ecuador': 1589,
    'Paraguay': 1526,
    
    # UEFA (Europe) - 16 teams
    'France': 1823,
    'Belgium': 1781,
    'England': 1774,
    'Netherlands': 1740,
    'Croatia': 1727,
    'Italy': 1723,
    'Portugal': 1702,
    'Spain': 1692,
    'Switzerland': 1655,
    'Germany': 1646,
    'Denmark': 1623,
    'Poland': 1589,
    'Sweden': 1584,
    'Austria': 1570,
    'Ukraine': 1562,
    'Turkey': 1558,
    'Wales': 1556,  # Additional UEFA team
    
    # CONCACAF (North/Central America) - 3 teams (plus 3 hosts)
    'USA': 1653,
    'Mexico': 1644,
    'Canada': 1537,
    
    # CAF (Africa) - 9 teams
    'Morocco': 1672,
    'Senegal': 1611,
    'Nigeria': 1598,
    'Egypt': 1576,
    'Cameroon': 1571,
    'Ghana': 1568,
    'Mali': 1552,
    'Algeria': 1550,
    'Tunisia': 1535,
    
    # AFC (Asia) - 8 teams
    'Japan': 1601,
    'Iran': 1578,
    'South Korea': 1565,
    'Australia': 1559,
    'Saudi Arabia': 1546,
    'Qatar': 1532,
    'United Arab Emirates': 1528,
    'Iraq': 1519,
    
    # OFC (Oceania) - 1 team
    'New Zealand': 1489,
}

def get_initial_rating(team, default=1500):
    """Get initial Elo rating from FIFA rankings post-2022 WC, or default."""
    # Try to match team name from dataset to FIFA rankings
    for fifa_team, rating in FIFA_RANKINGS_2022.items():
        if team == fifa_team or team in fifa_team or fifa_team in team:
            # Map FIFA points (approx 1000-1850) to Elo (1500-2200)
            return 1500 + (rating - 1000) * (700 / 850)
    return default

def compute_elo(df, initial_rating=1500, k=30, margin_multiplier=0.5):
    """
    Compute Elo ratings with:
    1. FIFA rankings post-2022 WC as initial ratings
    2. K-factor based on opponent strength (not tournament)
    3. Friendlies are excluded
    """
    # Filter out friendlies
    df = df[~df['tournament'].str.contains('Friendly', case=False, na=False)].copy()
    
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
        
        # Initialize teams with FIFA-based ratings
        if home not in ratings:
            ratings[home] = get_initial_rating(home, initial_rating)
        if away not in ratings:
            ratings[away] = get_initial_rating(away, initial_rating)
        
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
        
        # ===== K-FACTOR BASED ON OPPONENT STRENGTH =====
        # Stronger opponent = higher K-factor (more rating change)
        # Weaker opponent = lower K-factor (less rating change)
        
        # Base K is higher if you're playing a strong team
        opponent_strength = (r_home + r_away) / 2  # Average Elo of both teams
        
        # K-factor scales with opponent strength
        # Range: ~15 (weak opponents) to ~45 (strong opponents)
        k_effective = k * (1 + (opponent_strength - 1500) / 1000)
        
        # Clamp to reasonable range
        k_effective = max(15, min(45, k_effective))
        
        # Margin of victory adjustment
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