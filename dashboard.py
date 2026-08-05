import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from itertools import combinations
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.elo import compute_elo
from src.data_loader import load_data

# --- Define groups ---
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

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 15px; margin-bottom: 2rem; color: white; }
    .main-header h1 { font-size: 3rem; margin-bottom: 0; }
    .main-header p { font-size: 1.2rem; opacity: 0.8; }
    .bracket-match { background: #1e1e2f; border: 2px solid #333; border-radius: 8px; padding: 6px 10px; margin: 4px 0; font-size: 0.85rem; }
    .bracket-match.winner { border: 2px solid #f7c948; background: #2a2a3f; }
    .bracket-match.final { border: 2px solid #f7c948; background: #1a1a3f; }
    .bracket-match.third { border: 2px solid #cd7f32; background: #2a2a3f; }
    .champion-message { text-align: center; font-size: 2rem; font-weight: bold; color: #f7c948; padding: 1.5rem; margin: 1rem 0; background: linear-gradient(135deg, #1a1a2e, #2a2a4f); border-radius: 15px; border: 3px solid #f7c948; }
    .medal-stand { display: flex; justify-content: center; gap: 30px; margin: 20px 0 30px 0; flex-wrap: wrap; }
    .medal-item { text-align: center; background: #1e1e2f; border-radius: 15px; padding: 1.5rem 2rem; min-width: 150px; }
    .medal-item.gold { border: 3px solid #f7c948; }
    .medal-item.silver { border: 3px solid #c0c0c0; }
    .medal-item.bronze { border: 3px solid #cd7f32; }
    .medal-item .medal { font-size: 3rem; }
    .medal-item .team { font-size: 1.3rem; font-weight: bold; margin-top: 5px; }
    .medal-item .label { color: #888; font-size: 0.9rem; }
    .gold .team { color: #f7c948; }
    .silver .team { color: #c0c0c0; }
    .bronze .team { color: #cd7f32; }
</style>
""", unsafe_allow_html=True)

# --- Cached Loaders ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"Model not found. Error: {e}")
        return None, None, None

@st.cache_data
def load_elo_data():
    try:
        df = load_data()
        df = compute_elo(df)
        latest_elos = {}
        all_teams = set(df['home_team']).union(set(df['away_team']))
        for team in all_teams:
            home_elos = df[df['home_team'] == team]['home_elo_after']
            away_elos = df[df['away_team'] == team]['away_elo_after']
            if not home_elos.empty:
                latest_elos[team] = home_elos.iloc[-1]
            elif not away_elos.empty:
                latest_elos[team] = away_elos.iloc[-1]
        return latest_elos
    except Exception as e:
        st.error(f"Failed to load Elo data: {e}")
        return {}

# --- Simulation Functions ---
def predict_match_prob(home, away, elo_dict, model, scaler, feature_cols, neutral=True):
    home_elo = elo_dict.get(home, 1500)
    away_elo = elo_dict.get(away, 1500)
    features = pd.DataFrame([{
        'elo_diff': home_elo - away_elo,
        'elo_abs_diff': abs(home_elo - away_elo),
        'avg_elo': (home_elo + away_elo) / 2,
        'is_friendly': 0,
        'is_neutral': 1 if neutral else 0
    }])
    X_scaled = scaler.transform(features[feature_cols])
    probs = model.predict_proba(X_scaled)[0]
    return probs

def simulate_penalty_shootout():
    score1, score2 = 0, 0
    for _ in range(5):
        if np.random.random() < 0.72:
            score1 += 1
        if np.random.random() < 0.72:
            score2 += 1
    round_num = 6
    while score1 == score2:
        if np.random.random() < 0.72:
            score1 += 1
        if np.random.random() < 0.72:
            score2 += 1
        round_num += 1
    return score1, score2, round_num - 1

def simulate_match_score(home, away, elo_dict, model, scaler, feature_cols, neutral=True):
    probs = predict_match_prob(home, away, elo_dict, model, scaler, feature_cols, neutral)
    outcome = np.random.choice([0, 1, 2], p=probs)
    home_elo = elo_dict.get(home, 1500)
    away_elo = elo_dict.get(away, 1500)
    elo_diff = (home_elo - away_elo) / 400
    home_expected = 0.8 + 0.1 * elo_diff
    away_expected = 0.8 - 0.1 * elo_diff
    home_goals = np.random.poisson(max(0, home_expected + np.random.normal(0, 0.2)))
    away_goals = np.random.poisson(max(0, away_expected + np.random.normal(0, 0.2)))
    if outcome == 1:
        avg = (home_goals + away_goals) // 2
        home_goals, away_goals = avg, avg
    elif outcome == 2:
        if home_goals <= away_goals:
            home_goals = max(home_goals, away_goals + 1)
    else:
        if away_goals <= home_goals:
            away_goals = max(away_goals, home_goals + 1)
    return home_goals, away_goals, outcome

def simulate_match_with_penalty(home, away, elo_dict, model, scaler, feature_cols):
    hg, ag, out = simulate_match_score(home, away, elo_dict, model, scaler, feature_cols, neutral=True)
    if out == 2:
        return home, away, hg, ag, None, None, home
    elif out == 0:
        return home, away, hg, ag, None, None, away
    else:
        p1, p2, rounds = simulate_penalty_shootout()
        winner = home if p1 > p2 else away
        return home, away, hg, ag, f"{p1}-{p2}", rounds, winner

def simulate_group_stage(groups, elo_dict, model, scaler, feature_cols):
    results, standings = {}, {}
    for group_name, teams in groups.items():
        group_standings = {team: {'points': 0, 'gd': 0, 'gf': 0, 'ga': 0} for team in teams}
        group_results = []
        for home, away in combinations(teams, 2):
            hg, ag, out = simulate_match_score(home, away, elo_dict, model, scaler, feature_cols, neutral=False)
            group_results.append((home, away, hg, ag))
            if out == 2:
                group_standings[home]['points'] += 3
            elif out == 0:
                group_standings[away]['points'] += 3
            else:
                group_standings[home]['points'] += 1
                group_standings[away]['points'] += 1
            group_standings[home]['gf'] += hg
            group_standings[home]['ga'] += ag
            group_standings[home]['gd'] += hg - ag
            group_standings[away]['gf'] += ag
            group_standings[away]['ga'] += hg
            group_standings[away]['gd'] += ag - hg
        sorted_teams = sorted(group_standings.items(), key=lambda x: (-x[1]['points'], -x[1]['gd'], -x[1]['gf']))
        results[group_name] = group_results
        standings[group_name] = sorted_teams
    return results, standings

def generate_round_of_32(group_winners, group_runners):
    """Generate 16 Round of 32 matches from 32 qualified teams."""
    # Get all qualified teams (12 winners + 12 runners-up = 24 teams)
    # We need 32 teams, so we add the 8 best 3rd place teams (simulated)
    
    all_teams = []
    
    # Add group winners and runners-up
    for group in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if group in group_winners and group_winners[group]:
            all_teams.append(group_winners[group])
        if group in group_runners and group_runners[group]:
            all_teams.append(group_runners[group])
    
    # If we have less than 32 teams, add random teams from groups
    # This simulates the 8 best 3rd place teams
    all_group_teams = []
    for group in groups.values():
        all_group_teams.extend(group)
    
    # Remove teams already in all_teams
    remaining_teams = [t for t in all_group_teams if t not in all_teams]
    random.shuffle(remaining_teams)
    
    # Add 8 random teams to make 32
    needed = 32 - len(all_teams)
    all_teams.extend(remaining_teams[:needed])
    
    # Shuffle and pair
    random.shuffle(all_teams)
    
    matches = []
    for i in range(0, len(all_teams), 2):
        if i+1 < len(all_teams):
            matches.append((all_teams[i], all_teams[i+1]))
    
    return matches

def run_full_tournament(groups, elo_dict, model, scaler, feature_cols):
    # Group stage
    _, group_standings = simulate_group_stage(groups, elo_dict, model, scaler, feature_cols)
    
    group_winners = {}
    group_runners = {}
    for gn, standings in group_standings.items():
        if len(standings) > 0:
            group_winners[gn] = standings[0][0]
        if len(standings) > 1:
            group_runners[gn] = standings[1][0]
    
    # Round of 32
    r32_pairs = generate_round_of_32(group_winners, group_runners)
    r32_matches = []
    r32_winners = []
    for home, away in r32_pairs:
        result = simulate_match_with_penalty(home, away, elo_dict, model, scaler, feature_cols)
        r32_matches.append(result)
        r32_winners.append(result[6])
    
    # Round of 16
    r16_matches = []
    r16_winners = []
    for i in range(0, len(r32_winners), 2):
        if i+1 < len(r32_winners):
            result = simulate_match_with_penalty(r32_winners[i], r32_winners[i+1], elo_dict, model, scaler, feature_cols)
            r16_matches.append(result)
            r16_winners.append(result[6])
    
    # Quarter-finals
    qf_matches = []
    qf_winners = []
    for i in range(0, len(r16_winners), 2):
        if i+1 < len(r16_winners):
            result = simulate_match_with_penalty(r16_winners[i], r16_winners[i+1], elo_dict, model, scaler, feature_cols)
            qf_matches.append(result)
            qf_winners.append(result[6])
    
    # Semi-finals
    sf_matches = []
    sf_winners = []
    sf_teams = qf_winners[:4]
    for i in range(0, len(sf_teams), 2):
        if i+1 < len(sf_teams):
            result = simulate_match_with_penalty(sf_teams[i], sf_teams[i+1], elo_dict, model, scaler, feature_cols)
            sf_matches.append(result)
            sf_winners.append(result[6])
    
    # Third place match
    third_matches = []
    third_winner = None
    sf_losers = [t for t in sf_teams if t not in sf_winners]
    if len(sf_losers) >= 2:
        result = simulate_match_with_penalty(sf_losers[0], sf_losers[1], elo_dict, model, scaler, feature_cols)
        third_matches.append(result)
        third_winner = result[6]
    
    # Final
    final_matches = []
    champion = None
    runner_up = None
    if len(sf_winners) >= 2:
        result = simulate_match_with_penalty(sf_winners[0], sf_winners[1], elo_dict, model, scaler, feature_cols)
        final_matches.append(result)
        champion = result[6]
        runner_up = sf_winners[1] if sf_winners[0] == champion else sf_winners[0]
    
    return {
        'round32': r32_matches,
        'round16': r16_matches,
        'quarter': qf_matches,
        'semi': sf_matches,
        'third': third_matches,
        'final': final_matches,
        'champion': champion,
        'runner_up': runner_up,
        'third_place': third_winner
    }

def display_match(match):
    if len(match) >= 7:
        home, away, hg, ag, penalty, rounds, winner = match[:7]
    else:
        return
    
    if winner:
        st.markdown(f"<div class='bracket-match winner'><b>{home}</b> {hg}-{ag} <b>{away}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bracket-match'><b>{home}</b> {hg}-{ag} <b>{away}</b></div>", unsafe_allow_html=True)
    
    if penalty:
        st.markdown(f"<div style='font-size:0.7rem;color:#aaa;text-align:center;'>Penalties: {penalty}</div>", unsafe_allow_html=True)

# --- Monte Carlo ---
@st.cache_data(ttl=3600)
def run_monte_carlo(groups_dict, elo_dict, num_simulations):
    model, scaler, feature_cols = load_model()
    if model is None:
        return {}
    
    results = {}
    for _ in range(num_simulations):
        _, group_standings = simulate_group_stage(groups_dict, elo_dict, model, scaler, feature_cols)
        group_winners = {}
        group_runners = {}
        for gn, standings in group_standings.items():
            if len(standings) > 0:
                group_winners[gn] = standings[0][0]
            if len(standings) > 1:
                group_runners[gn] = standings[1][0]
        
        r32_pairs = generate_round_of_32(group_winners, group_runners)
        r32_winners = []
        for home, away in r32_pairs:
            result = simulate_match_with_penalty(home, away, elo_dict, model, scaler, feature_cols)
            r32_winners.append(result[6])
        
        r16_winners = []
        for i in range(0, len(r32_winners), 2):
            if i+1 < len(r32_winners):
                result = simulate_match_with_penalty(r32_winners[i], r32_winners[i+1], elo_dict, model, scaler, feature_cols)
                r16_winners.append(result[6])
        
        qf_winners = []
        for i in range(0, len(r16_winners), 2):
            if i+1 < len(r16_winners):
                result = simulate_match_with_penalty(r16_winners[i], r16_winners[i+1], elo_dict, model, scaler, feature_cols)
                qf_winners.append(result[6])
        
        sf_winners = []
        sf_teams = qf_winners[:4]
        for i in range(0, len(sf_teams), 2):
            if i+1 < len(sf_teams):
                result = simulate_match_with_penalty(sf_teams[i], sf_teams[i+1], elo_dict, model, scaler, feature_cols)
                sf_winners.append(result[6])
        
        if len(sf_winners) >= 2:
            result = simulate_match_with_penalty(sf_winners[0], sf_winners[1], elo_dict, model, scaler, feature_cols)
            winner = result[6]
            results[winner] = results.get(winner, 0) + 1
    
    return results

# --- UI ---
st.markdown('<div class="main-header"><h1>🏆 FIFA World Cup 2026</h1><p>Predictor & Simulation Engine</p></div>', unsafe_allow_html=True)

model, scaler, feature_cols = load_model()
elo_dict = load_elo_data()

if model is None or not elo_dict:
    st.stop()

with st.sidebar:
    st.markdown("⚙️ Controls")
    st.markdown(f"**Teams:** 48\n**Groups:** 12\n**Model:** XGBoost + Elo")
    if st.button("🔄 New Simulation", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Groups", "🏆 Knockout", "🎯 Win Probabilities", "📖 Match Details"])

# Tab 1: Groups
with tab1:
    st.subheader("Group Stage Standings")
    selected_group = st.selectbox("Select Group", list(groups.keys()))
    group_results, group_standings = simulate_group_stage({selected_group: groups[selected_group]}, elo_dict, model, scaler, feature_cols)
    
    st.markdown("### Matches")
    for match in group_results[selected_group]:
        home, away, hg, ag = match
        c1, c2, c3 = st.columns([4, 1, 4])
        c1.markdown(f"**{home}**")
        c2.markdown(f"{hg} - {ag}")
        c3.markdown(f"**{away}**")
    
    st.markdown("### Standings")
    h1, h2, h3, h4, h5, h6 = st.columns([2, 1, 1, 1, 1, 1])
    h1.markdown("**Team**")
    h2.markdown("**Pts**")
    h3.markdown("**GD**")
    h4.markdown("**GF**")
    h5.markdown("**GA**")
    h6.markdown("**Pld**")
    
    for pos, (team, stats) in enumerate(group_standings[selected_group], 1):
        c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
        c1.markdown(f"{pos}. {team}")
        c2.markdown(f"{stats['points']}")
        c3.markdown(f"{stats['gd']}")
        c4.markdown(f"{stats['gf']}")
        c5.markdown(f"{stats['ga']}")
        c6.markdown(f"3")

# Tab 2: Knockout
with tab2:
    st.subheader("Knockout Stage Bracket")
    
    data = run_full_tournament(groups, elo_dict, model, scaler, feature_cols)
    
    # Medal Stand (Top 3 ABOVE bracket)
    st.markdown("### 🏅 Final Standings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data['champion']:
            st.markdown(f"""
            <div class="medal-item gold">
                <div class="medal">🥇</div>
                <div class="team">{data['champion']}</div>
                <div class="label">Champion</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if data['runner_up']:
            st.markdown(f"""
            <div class="medal-item silver">
                <div class="medal">🥈</div>
                <div class="team">{data['runner_up']}</div>
                <div class="label">Runner-up</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if data['third_place']:
            st.markdown(f"""
            <div class="medal-item bronze">
                <div class="medal">🥉</div>
                <div class="team">{data['third_place']}</div>
                <div class="label">Third Place</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Bracket
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown("<div style='text-align:center;font-weight:bold;color:#f7c948;margin-bottom:10px;'>Round of 32</div>", unsafe_allow_html=True)
        if data['round32']:
            for match in data['round32']:
                display_match(match)
        else:
            st.markdown("<div style='color:#555;text-align:center;'>No matches</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='text-align:center;font-weight:bold;color:#f7c948;margin-bottom:10px;'>Round of 16</div>", unsafe_allow_html=True)
        if data['round16']:
            for match in data['round16']:
                display_match(match)
        else:
            st.markdown("<div style='color:#555;text-align:center;'>No matches</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div style='text-align:center;font-weight:bold;color:#f7c948;margin-bottom:10px;'>Quarter-finals</div>", unsafe_allow_html=True)
        if data['quarter']:
            for match in data['quarter']:
                display_match(match)
        else:
            st.markdown("<div style='color:#555;text-align:center;'>No matches</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div style='text-align:center;font-weight:bold;color:#f7c948;margin-bottom:10px;'>Semi-finals</div>", unsafe_allow_html=True)
        if data['semi']:
            for match in data['semi']:
                display_match(match)
        else:
            st.markdown("<div style='color:#555;text-align:center;'>No matches</div>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("<div style='text-align:center;font-weight:bold;color:#cd7f32;margin-bottom:10px;'>Third Place</div>", unsafe_allow_html=True)
        if data['third']:
            for match in data['third']:
                display_match(match)
        else:
            st.markdown("<div style='color:#555;text-align:center;'>No match</div>", unsafe_allow_html=True)
    
    with col6:
        st.markdown("<div style='text-align:center;font-weight:bold;color:#f7c948;margin-bottom:10px;'>Final</div>", unsafe_allow_html=True)
        if data['final']:
            for match in data['final']:
                home, away, hg, ag, penalty, rounds, winner = match[:7]
                st.markdown(f"<div class='bracket-match final'><b>{home}</b> {hg}-{ag} <b>{away}</b></div>", unsafe_allow_html=True)
                if penalty:
                    st.markdown(f"<div style='font-size:0.7rem;color:#aaa;text-align:center;'>Penalties: {penalty}</div>", unsafe_allow_html=True)
                if winner:
                    st.markdown(f"<div style='text-align:center;color:#f7c948;font-weight:bold;'>🏆 {winner}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#555;text-align:center;'>TBD</div>", unsafe_allow_html=True)
    
    if data['champion']:
        st.markdown(f"""
        <div class="champion-message">
            <div>🏆</div>
            {data['champion']} is FIFA World Cup 2026 Champion!
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Win Probabilities
with tab3:
    st.subheader("Win Probabilities")
    
    if 'mc_results' not in st.session_state:
        st.session_state['mc_results'] = None
    if 'mc_runs' not in st.session_state:
        st.session_state['mc_runs'] = 0
    
    num_sims = st.slider("Number of simulations", 100, 5000, 1000, step=100)
    
    if st.button("🏃 Run Monte Carlo", type="primary", use_container_width=True):
        with st.spinner(f"Running {num_sims} simulations..."):
            results = run_monte_carlo(groups, elo_dict, num_sims)
            st.session_state['mc_results'] = results
            st.session_state['mc_runs'] = num_sims
    
    if st.session_state['mc_results']:
        results = st.session_state['mc_results']
        runs = st.session_state['mc_runs']
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        st.markdown("### Win Probability Distribution")
        for team, wins in sorted_results[:20]:
            prob = (wins / runs) * 100
            st.markdown(f"""
            <div style="margin: 4px 0;">
                <span style="display: inline-block; width: 140px; font-weight: bold;">{team}</span>
                <span style="display: inline-block; width: 60px; text-align: right; font-weight: bold;">{prob:.1f}%</span>
                <span style="display: inline-block; height: 22px; background: linear-gradient(90deg, #f7c948, #f5a623); border-radius: 10px; width: {min(prob * 2.5, 300)}px;"></span>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Teams Simulated", len(results))
        with col2:
            st.metric("Total Runs", runs)
        with col3:
            if sorted_results:
                st.metric("🏆 Leader", sorted_results[0][0])
    else:
        st.info("Click 'Run Monte Carlo' to see win probabilities.")

# Tab 4: Match Details
with tab4:
    st.subheader("Match Predictions")
    team_list = sorted(list(elo_dict.keys()))
    
    c1, c2 = st.columns(2)
    with c1:
        team1 = st.selectbox("Team 1", team_list, key='t1')
    with c2:
        team2 = st.selectbox("Team 2", team_list, key='t2')
    
    if team1 and team2 and team1 != team2:
        probs = predict_match_prob(team1, team2, elo_dict, model, scaler, feature_cols, neutral=True)
        
        st.markdown("### Match Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{team1} Win", f"{probs[2]*100:.1f}%")
        col2.metric("Draw", f"{probs[1]*100:.1f}%")
        col3.metric(f"{team2} Win", f"{probs[0]*100:.1f}%")
        
        st.divider()
        st.markdown("### Simulated Score")
        hg, ag, _ = simulate_match_score(team1, team2, elo_dict, model, scaler, feature_cols, neutral=True)
        col1, col2, col3 = st.columns([4, 2, 4])
        col1.markdown(f"## {team1}")
        col2.markdown(f"## {hg} - {ag}")
        col3.markdown(f"## {team2}")
    else:
        st.warning("Please select two different teams.")