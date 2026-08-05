from data_loader import load_data
from elo import compute_elo
from features import create_features

# Load data
df = load_data()

# Compute Elo
df = compute_elo(df)

# Create features
df = create_features(df)

# Check the columns
print("📊 Feature columns:")
print(df.columns.tolist())

# Check a few rows
print("\n📊 Sample data:")
print(df[['date', 'home_team', 'away_team', 'home_elo', 'away_elo', 'elo_diff', 'result']].head(10))

# Check distribution of results
print("\n📊 Result distribution:")
print(df['result'].value_counts())