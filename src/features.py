import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature columns for each match.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # 1. Elo-based features
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['elo_abs_diff'] = abs(df['elo_diff'])
    df['avg_elo'] = (df['home_elo'] + df['away_elo']) / 2
    
    # 2. Is the match a friendly?
    df['is_friendly'] = df['tournament'].str.contains('Friendly', case=False, na=False).astype(int)
    
    # 3. Is the match at a neutral venue?
    df['is_neutral'] = df['neutral'].astype(int)
    
    # 4. Goal difference (for training only, not used for prediction)
    df['goal_diff'] = df['home_score'] - df['away_score']
    
    # 5. Target variable: result (0 = away win, 1 = draw, 2 = home win)
    df['result'] = 0  # away win
    df.loc[df['home_score'] == df['away_score'], 'result'] = 1  # draw
    df.loc[df['home_score'] > df['away_score'], 'result'] = 2  # home win
    
    df = df[df['date'] > '2022-12-18']
    return df
