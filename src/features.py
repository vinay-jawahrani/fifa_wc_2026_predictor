import pandas as pd
import numpy as np

def compute_recent_form(df, team, date, matches=5):
    """
    Compute recent form (points per game) for a team in last N matches.
    No home/away bias – treats all matches equally.
    """
    # Get matches before the given date
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(matches)
    
    if len(team_matches) == 0:
        return 1.5  # Neutral default
    
    points = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            # Team is home
            if row['home_score'] > row['away_score']:
                points += 3
            elif row['home_score'] == row['away_score']:
                points += 1
        else:
            # Team is away
            if row['away_score'] > row['home_score']:
                points += 3
            elif row['home_score'] == row['away_score']:
                points += 1
    
    return points / len(team_matches)


def compute_goal_diff_avg(df, team, date, matches=5):
    """
    Compute average goal difference in last N matches.
    No home/away bias – treats all matches equally.
    """
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(matches)
    
    if len(team_matches) == 0:
        return 0
    
    goal_diff_sum = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            goal_diff_sum += row['home_score'] - row['away_score']
        else:
            goal_diff_sum += row['away_score'] - row['home_score']
    
    return goal_diff_sum / len(team_matches)


def compute_recent_form_weighted(df, team, date, matches=10):
    """
    Compute recent form with exponential weighting (recent matches matter more).
    No home/away bias – treats all matches equally.
    """
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(matches)
    
    if len(team_matches) == 0:
        return 1.5
    
    # Exponential weights: more recent = higher weight
    weights = np.exp(np.linspace(0, 1, len(team_matches)))
    weights = weights / weights.sum()
    
    points = 0
    for i, (_, row) in enumerate(team_matches.iterrows()):
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']:
                points += 3 * weights[i]
            elif row['home_score'] == row['away_score']:
                points += 1 * weights[i]
        else:
            if row['away_score'] > row['home_score']:
                points += 3 * weights[i]
            elif row['home_score'] == row['away_score']:
                points += 1 * weights[i]
    
    return points

# In features.py, add goal difference rolling average
def compute_goal_diff_avg(df, team, date, matches=5):
    recent = df[(df['home_team'] == team) | (df['away_team'] == team)].tail(matches)
    goal_diff = 0
    for _, row in recent.iterrows():
        if row['home_team'] == team:
            goal_diff += row['home_score'] - row['away_score']
        else:
            goal_diff += row['away_score'] - row['home_score']
    return goal_diff / matches if matches > 0 else 0

def create_features(df):
    df = df.copy()
    
    # Elo-based
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['elo_abs_diff'] = abs(df['elo_diff'])
    df['avg_elo'] = (df['home_elo'] + df['away_elo']) / 2
    
    # Form (no home bias)
    df['home_form'] = df.apply(lambda row: compute_recent_form(df, row['home_team'], row['date'], 5), axis=1)
    df['away_form'] = df.apply(lambda row: compute_recent_form(df, row['away_team'], row['date'], 5), axis=1)
    
    # Goal difference average (no home bias)
    df['home_gd_avg'] = df.apply(lambda row: compute_goal_diff_avg(df, row['home_team'], row['date'], 5), axis=1)
    df['away_gd_avg'] = df.apply(lambda row: compute_goal_diff_avg(df, row['away_team'], row['date'], 5), axis=1)
    
    # Existing features
    df['is_friendly'] = df['tournament'].str.contains('Friendly', case=False, na=False).astype(int)
    df['is_neutral'] = df['neutral'].astype(int)
    df['goal_diff'] = df['home_score'] - df['away_score']
    df['result'] = 0
    df.loc[df['home_score'] == df['away_score'], 'result'] = 1
    df.loc[df['home_score'] > df['away_score'], 'result'] = 2
    
    return df