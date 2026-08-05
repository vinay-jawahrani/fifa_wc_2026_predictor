import pandas as pd
import numpy as np
import joblib
from itertools import combinations

# Load the trained model and scaler
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')

# --- 2026 World Cup Groups (Official) ---
groups = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['USA', 'Paraguay', 'Australia', 'Türkiye'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}

# --- Shared Functions (used by both simulation and Monte Carlo) ---

def get_elo_for_team(team, elo_dict):
    """Get Elo rating for a team, with a default if missing."""
    return elo_dict.get(team, 1500)

def predict_match(home_team, away_team, elo_dict, neutral=True):
    """Predict match outcome probabilities."""
    home_elo = get_elo_for_team(home_team, elo_dict)
    away_elo = get_elo_for_team(away_team, elo_dict)
    
    features = pd.DataFrame([{
        'elo_diff': home_elo - away_elo,
        'elo_abs_diff': abs(home_elo - away_elo),
        'avg_elo': (home_elo + away_elo) / 2,
        'is_friendly': 0,
        'is_neutral': 1 if neutral else 0
    }])
    
    X_scaled = scaler.transform(features[feature_cols])
    probs = model.predict_proba(X_scaled)[0]  # [away, draw, home]
    return probs

def simulate_match(probs):
    """Simulate a single match outcome."""
    return np.random.choice([0, 1, 2], p=probs)  # 0=away win, 1=draw, 2=home win

def simulate_group(group_teams, elo_dict):
    """Simulate a group and return standings as a dictionary."""
    standings = {team: 0 for team in group_teams}
    
    for home, away in combinations(group_teams, 2):
        probs = predict_match(home, away, elo_dict, neutral=False)
        outcome = simulate_match(probs)
        
        if outcome == 2:  # home win
            standings[home] += 3
        elif outcome == 0:  # away win
            standings[away] += 3
        else:  # draw
            standings[home] += 1
            standings[away] += 1
    
    return standings

def simulate_knockout(teams, elo_dict):
    """Simulate a knockout round and return winners."""
    if len(teams) < 2:
        return teams
    
    winners = []
    for i in range(0, len(teams) - 1, 2):
        home = teams[i]
        away = teams[i+1]
        probs = predict_match(home, away, elo_dict, neutral=True)
        outcome = simulate_match(probs)
        
        if outcome == 2:  # home win
            winners.append(home)
        elif outcome == 0:  # away win
            winners.append(away)
        else:  # draw -> random winner
            winners.append(np.random.choice([home, away]))
    
    if len(teams) % 2 == 1:
        winners.append(teams[-1])
    
    return winners

def simulate_tournament(elo_dict, num_simulations=10000):
    """Run Monte Carlo simulation for win probabilities."""
    results = {team: 0 for group in groups.values() for team in group}
    
    for sim in range(num_simulations):
        group_winners = []
        best_third = []
        
        for group_teams in groups.values():
            standings = simulate_group(group_teams, elo_dict)
            sorted_teams = sorted(standings.items(), key=lambda x: x[1], reverse=True)
            group_winners.extend([t[0] for t in sorted_teams[:2]])
            best_third.append(sorted_teams[2][0])
        
        all_32 = group_winners.copy()
        np.random.shuffle(best_third)
        all_32.extend(best_third[:8])
        
        if len(all_32) < 32:
            remaining = [t for group in groups.values() for t in group if t not in all_32]
            np.random.shuffle(remaining)
            all_32.extend(remaining[:32 - len(all_32)])
        
        round_32 = simulate_knockout(all_32, elo_dict)
        round_16 = simulate_knockout(round_32, elo_dict)
        quarter = simulate_knockout(round_16, elo_dict)
        semi = simulate_knockout(quarter, elo_dict)
        final = simulate_knockout(semi, elo_dict)
        
        if final:
            results[final[0]] += 1
    
    return results

# --- Functions for Detailed Tournament Display ---

def predict_match_with_score(home_team, away_team, elo_dict, neutral=True):
    """Predict match outcome with probabilities and simulate actual score."""
    home_elo = get_elo_for_team(home_team, elo_dict)
    away_elo = get_elo_for_team(away_team, elo_dict)
    
    features = pd.DataFrame([{
        'elo_diff': home_elo - away_elo,
        'elo_abs_diff': abs(home_elo - away_elo),
        'avg_elo': (home_elo + away_elo) / 2,
        'is_friendly': 0,
        'is_neutral': 1 if neutral else 0
    }])
    
    X_scaled = scaler.transform(features[feature_cols])
    probs = model.predict_proba(X_scaled)[0]
    
    outcome = np.random.choice([0, 1, 2], p=probs)
    
    elo_diff = (home_elo - away_elo) / 400
    home_expected = 0.8 + 0.1 * elo_diff
    away_expected = 0.8 - 0.1 * elo_diff
    
    home_goals = np.random.poisson(max(0, home_expected + np.random.normal(0, 0.2)))
    away_goals = np.random.poisson(max(0, away_expected + np.random.normal(0, 0.2)))
    
    if outcome == 1:
        if home_goals != away_goals:
            avg = (home_goals + away_goals) // 2
            home_goals, away_goals = avg, avg
    elif outcome == 2:
        if home_goals <= away_goals:
            home_goals = max(home_goals, away_goals + 1)
    else:
        if away_goals <= home_goals:
            away_goals = max(away_goals, home_goals + 1)
    
    return home_goals, away_goals, outcome

def simulate_penalty_shootout(team1, team2):
    """
    Simulate a penalty shootout between two teams.
    Returns: (winner, score_line)
    """
    # Conversion rates: 70-75% is realistic for professional players
    conversion_rate1 = 0.70 + np.random.random() * 0.05
    conversion_rate2 = 0.70 + np.random.random() * 0.05
    
    score1 = 0
    score2 = 0
    
    # 5 rounds
    for _ in range(5):
        if np.random.random() < conversion_rate1:
            score1 += 1
        if np.random.random() < conversion_rate2:
            score2 += 1
    
    # Sudden death if tied
    round_num = 6
    while score1 == score2:
        if np.random.random() < conversion_rate1:
            score1 += 1
        if np.random.random() < conversion_rate2:
            score2 += 1
        round_num += 1
    
    winner = team1 if score1 > score2 else team2
    return winner, f"{score1}-{score2} (after {round_num - 1} rounds)"

def simulate_group_with_scores(group_teams, elo_dict):
    """Simulate a group and return standings with full results."""
    standings = {team: {'points': 0, 'gd': 0, 'gf': 0, 'ga': 0} for team in group_teams}
    results = {}
    
    for home, away in combinations(group_teams, 2):
        home_goals, away_goals, outcome = predict_match_with_score(home, away, elo_dict, neutral=False)
        results[f"{home} vs {away}"] = f"{home_goals} - {away_goals}"
        
        if outcome == 2:
            standings[home]['points'] += 3
        elif outcome == 0:
            standings[away]['points'] += 3
        else:
            standings[home]['points'] += 1
            standings[away]['points'] += 1
        
        standings[home]['gf'] += home_goals
        standings[home]['ga'] += away_goals
        standings[home]['gd'] += home_goals - away_goals
        standings[away]['gf'] += away_goals
        standings[away]['ga'] += home_goals
        standings[away]['gd'] += away_goals - home_goals
    
    return standings, results

def simulate_knockout_with_scores(teams, elo_dict):
    """Simulate a knockout round with scores and penalty shootouts."""
    if len(teams) < 2:
        return teams, {}
    
    winners = []
    match_results = {}
    
    for i in range(0, len(teams) - 1, 2):
        home = teams[i]
        away = teams[i+1]
        home_goals, away_goals, outcome = predict_match_with_score(home, away, elo_dict, neutral=True)
        
        match_label = f"{home} vs {away}"
        
        if outcome == 2:  # home win
            winners.append(home)
            match_results[match_label] = f"{home_goals} - {away_goals}"
        elif outcome == 0:  # away win
            winners.append(away)
            match_results[match_label] = f"{home_goals} - {away_goals}"
        else:  # draw -> penalty shootout
            winner, penalty_score = simulate_penalty_shootout(home, away)
            winners.append(winner)
            match_results[match_label] = f"{home_goals} - {away_goals} (Penalties: {penalty_score})"
    
    if len(teams) % 2 == 1:
        winners.append(teams[-1])
        match_results[f"{teams[-1]} (Bye)"] = "Advanced without playing"
    
    return winners, match_results

def print_group_stage(groups, elo_dict):
    """Print full group stage results with standings."""
    print("\n" + "="*60)
    print("🏆 GROUP STAGE RESULTS")
    print("="*60)
    
    all_group_winners = []
    best_third = []
    
    for group_name, group_teams in groups.items():
        print(f"\n📌 Group {group_name}:")
        print("-"*40)
        
        standings, results = simulate_group_with_scores(group_teams, elo_dict)
        
        for match, score in results.items():
            print(f"  {match}: {score}")
        
        print("\n  Standings:")
        sorted_teams = sorted(standings.items(), key=lambda x: (-x[1]['points'], -x[1]['gd'], -x[1]['gf']))
        for pos, (team, stats) in enumerate(sorted_teams, 1):
            print(f"    {pos}. {team}: {stats['points']} pts, GD: {stats['gd']}, GF: {stats['gf']}, GA: {stats['ga']}")
            if pos <= 2:
                all_group_winners.append(team)
            if pos == 3:
                best_third.append(team)
    
    return all_group_winners, best_third

def print_knockout_round(teams, elo_dict, round_name):
    """Print a knockout round and return winners."""
    print(f"\n{'='*60}")
    print(f"🏆 {round_name.upper()}")
    print("="*60)
    
    winners, match_results = simulate_knockout_with_scores(teams, elo_dict)
    
    for match, score in match_results.items():
        print(f"  {match}: {score}")
    
    return winners

def main():
    print("🏆 Running World Cup 2026 simulation...")
    
    from elo import compute_elo
    from data_loader import load_data
    
    df = load_data()
    df = compute_elo(df, initial_rating=1500, k=30, margin_multiplier=0.5)
    
    latest_elos = {}
    all_teams = set(df['home_team']).union(set(df['away_team']))
    for team in all_teams:
        home_elos = df[df['home_team'] == team]['home_elo_after']
        away_elos = df[df['away_team'] == team]['away_elo_after']
        if not home_elos.empty:
            latest_elos[team] = home_elos.iloc[-1]
        elif not away_elos.empty:
            latest_elos[team] = away_elos.iloc[-1]
    
    print(f"📊 Loaded Elo ratings for {len(latest_elos)} teams")
    
    print("\n🏟️ SIMULATING FULL TOURNAMENT")
    print("="*60)
    
    group_winners, best_third = print_group_stage(groups, latest_elos)
    
    all_32 = group_winners.copy()
    np.random.shuffle(best_third)
    all_32.extend(best_third[:8])
    
    if len(all_32) < 32:
        remaining = [t for group in groups.values() for t in group if t not in all_32]
        np.random.shuffle(remaining)
        all_32.extend(remaining[:32 - len(all_32)])
    
    round_32_winners = print_knockout_round(all_32, latest_elos, "Round of 32")
    round_16_winners = print_knockout_round(round_32_winners, latest_elos, "Round of 16")
    quarter_winners = print_knockout_round(round_16_winners, latest_elos, "Quarter-finals")
    semi_winners = print_knockout_round(quarter_winners, latest_elos, "Semi-finals")
    final_winners = print_knockout_round(semi_winners, latest_elos, "Final")
    
    print(f"\n{'='*60}")
    print(f"🏆 2026 WORLD CUP WINNER: {final_winners[0] if final_winners else 'Unknown'}")
    print("="*60)
    
    print("\n" + "="*60)
    print("📊 MONTE CARLO SIMULATION (10,000 runs)")
    print("="*60)
    
    results = simulate_tournament(latest_elos, num_simulations=10000)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\n🏆 World Cup Win Probabilities:")
    for team, wins in sorted_results[:10]:
        prob = (wins / 10000) * 100
        print(f"  {team}: {prob:.1f}%")
    
    pd.DataFrame(sorted_results, columns=['Team', 'Wins']).to_csv(
        'data/processed/2026_world_cup_predictions.csv', index=False
    )
    print("\n✅ Predictions saved to 'data/processed/2026_world_cup_predictions.csv'")

if __name__ == "__main__":
    main()