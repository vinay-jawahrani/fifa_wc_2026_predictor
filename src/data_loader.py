import pandas as pd
from pathlib import Path

def load_data():
    """
    Load the international football results dataset.
    Expects 'results.csv' in data/raw/
    """
    # Define the path to the raw data
    raw_path = Path('data/raw/results.csv')
    
    # Check if the file exists
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found at {raw_path}. Please download it and place it in data/raw/")
    
    # Load the CSV
    df = pd.read_csv(raw_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date (oldest to newest)
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def explore_data(df):
    """
    Print basic statistics and information about the dataset.
    """
    print("=" * 50)
    print("📊 DATASET OVERVIEW")
    print("=" * 50)
    
    print(f"\n📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"📝 Total matches: {len(df)}")
    
    # Unique teams
    home_teams = set(df['home_team'].unique())
    away_teams = set(df['away_team'].unique())
    all_teams = home_teams.union(away_teams)
    print(f"🏆 Total teams: {len(all_teams)}")
    
    # Tournaments
    print(f"🏟️ Total tournaments: {df['tournament'].nunique()}")
    print(f"   Most common tournaments:\n{df['tournament'].value_counts().head(5)}")
    
    # Outcome distribution
    home_wins = (df['home_score'] > df['away_score']).sum()
    draws = (df['home_score'] == df['away_score']).sum()
    away_wins = (df['home_score'] < df['away_score']).sum()
    
    print("\n📈 Match Outcome Distribution:")
    print(f"   Home wins: {home_wins} ({home_wins/len(df)*100:.1f}%)")
    print(f"   Draws: {draws} ({draws/len(df)*100:.1f}%)")
    print(f"   Away wins: {away_wins} ({away_wins/len(df)*100:.1f}%)")
    
    # Check for recent matches (post 2022)
    recent = df[df['date'] > '2023-01-01']
    print(f"\n🆕 Recent matches (2023 onwards): {len(recent)}")

if __name__ == "__main__":
    try:
        df = load_data()
        explore_data(df)
        print("\n✅ Dataset loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")