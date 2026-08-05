from data_loader import load_data
from elo import compute_elo

# Load data
df = load_data()

# Compute Elo with enhancements
df = compute_elo(
    df, 
    initial_rating=1500, 
    k=30, 
    margin_multiplier=0.5,
    friendly_reduction=0.5
)

# Check top teams
latest_elos = {}
for team in set(df['home_team']).union(set(df['away_team'])):
    home_elos = df[df['home_team'] == team]['home_elo']
    away_elos = df[df['away_team'] == team]['away_elo']
    
    if not home_elos.empty:
        latest_elos[team] = home_elos.iloc[-1]
    elif not away_elos.empty:
        latest_elos[team] = away_elos.iloc[-1]

print("\n🏆 Top 10 Elo ratings:")
sorted_teams = sorted(latest_elos.items(), key=lambda x: x[1], reverse=True)[:10]
for team, elo in sorted_teams:
    print(f"  {team}: {elo:.1f}")