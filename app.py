import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="VEX Tournament Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .team-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .prediction-high {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .prediction-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class VEXTournamentAnalyzer:
    def __init__(self, data_file: str):
        """Initialize the VEX Tournament Analyzer with data from Excel file."""
        self.data_file = data_file
        self.df = None
        self.ranking_model = None
        self.match_model = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the tournament data."""
        try:
            self.df = pd.read_excel(self.data_file)
            # Clean and preprocess data
            self.df = self.df.fillna(0)
            # Ensure numeric columns are properly typed
            numeric_columns = ['tsRanking', 'trueskill', 'ccwm', 'totalWins', 'totalLosses', 
                             'totalTies', 'totalMatches', 'totalWinningPercent', 'eliminationWins',
                             'eliminationLosses', 'eliminationTies', 'eliminationWinningPercent',
                             'qualWins', 'qualLosses', 'qualTies', 'qualWinningPercent',
                             'apPerMatch', 'awpPerMatch', 'wpPerMatch', 'mu', 'tsSigma',
                             'opr', 'dpr', 'totalSkillsRanking', 'regionGradeSkillsRanking',
                             'scoreDriverMax', 'scoreAutoMax', 'scoreTotalMax']
            
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            
            st.success(f"✅ Data loaded successfully! {len(self.df)} teams found.")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            
    def calculate_composite_score(self, team_data: pd.Series) -> float:
        """Calculate a composite score for team ranking based on multiple factors."""
        # Weighted scoring system
        weights = {
            'trueskill': 0.25,
            'ccwm': 0.20,
            'totalWinningPercent': 0.15,
            'eliminationWinningPercent': 0.15,
            'opr': 0.10,
            'dpr': 0.05,  # Lower DPR is better, so we'll invert this
            'apPerMatch': 0.05,
            'awpPerMatch': 0.05
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in team_data and not pd.isna(team_data[metric]):
                if metric == 'dpr':
                    # Lower DPR is better, so invert it
                    score += weight * (100 - team_data[metric])  # Assuming max DPR around 100
                else:
                    score += weight * team_data[metric]
        
        return score
    
    def get_team_rankings(self) -> pd.DataFrame:
        """Generate comprehensive team rankings."""
        if self.df is None:
            return pd.DataFrame()
        
        # Calculate composite scores
        self.df['composite_score'] = self.df.apply(self.calculate_composite_score, axis=1)
        
        # Sort by composite score (descending)
        rankings = self.df.sort_values('composite_score', ascending=False).copy()
        rankings['custom_ranking'] = range(1, len(rankings) + 1)
        
        return rankings[['custom_ranking', 'teamNumber', 'teamName', 'composite_score', 
                        'trueskill', 'ccwm', 'totalWinningPercent', 'eliminationWinningPercent',
                        'opr', 'dpr', 'apPerMatch', 'awpPerMatch']]
    
    def predict_match_outcome(self, alliance1: List[str], alliance2: List[str]) -> Dict:
        """Predict the outcome of a 2v2 match between two alliances."""
        if len(alliance1) != 2 or len(alliance2) != 2:
            return {"error": "Each alliance must have exactly 2 teams"}
        
        # Get team data
        teams1 = []
        teams2 = []
        
        for team_num in alliance1:
            team = self.df[self.df['teamNumber'] == team_num]
            if team.empty:
                return {"error": f"Team {team_num} not found"}
            teams1.append(team.iloc[0])
        
        for team_num in alliance2:
            team = self.df[self.df['teamNumber'] == team_num]
            if team.empty:
                return {"error": f"Team {team_num} not found"}
            teams2.append(team.iloc[0])
        
        # Calculate alliance strengths
        alliance1_strength = sum(self.calculate_composite_score(team) for team in teams1)
        alliance2_strength = sum(self.calculate_composite_score(team) for team in teams2)
        
        # Add synergy bonus for complementary skills
        alliance1_synergy = self.calculate_alliance_synergy(teams1)
        alliance2_synergy = self.calculate_alliance_synergy(teams2)
        
        alliance1_total = alliance1_strength + alliance1_synergy
        alliance2_total = alliance2_strength + alliance2_synergy
        
        # Predict outcome
        strength_diff = alliance1_total - alliance2_total
        alliance1_win_prob = 1 / (1 + np.exp(-strength_diff / 15))  # Adjusted for 2v2
        alliance2_win_prob = 1 - alliance1_win_prob
        
        return {
            "alliance1": {
                "teams": alliance1,
                "team_names": [team['teamName'] for team in teams1],
                "strength": alliance1_strength,
                "synergy": alliance1_synergy,
                "total_strength": alliance1_total,
                "win_probability": alliance1_win_prob
            },
            "alliance2": {
                "teams": alliance2,
                "team_names": [team['teamName'] for team in teams2],
                "strength": alliance2_strength,
                "synergy": alliance2_synergy,
                "total_strength": alliance2_total,
                "win_probability": alliance2_win_prob
            },
            "predicted_winner": alliance1 if alliance1_win_prob > 0.5 else alliance2
        }
    
    def calculate_alliance_synergy(self, teams: List[pd.Series]) -> float:
        """Calculate synergy bonus for a 2-team alliance."""
        if len(teams) != 2:
            return 0
        
        team1, team2 = teams
        
        # Synergy factors for VEX V5
        synergy = 0
        
        # Complementary scoring abilities
        synergy += min(team1['opr'], team2['opr']) * 0.1  # Both teams can score
        
        # Defensive coordination
        synergy += (100 - max(team1['dpr'], team2['dpr'])) * 0.05  # Better defense together
        
        # Autonomous coordination
        synergy += (team1['apPerMatch'] + team2['apPerMatch']) * 0.2  # Combined autonomous
        
        # Win point coordination
        synergy += (team1['awpPerMatch'] + team2['awpPerMatch']) * 0.3  # Combined win points
        
        # Penalty for skill gaps (teams too far apart in skill)
        skill_diff = abs(team1['trueskill'] - team2['trueskill'])
        synergy -= skill_diff * 0.1  # Penalty for large skill gaps
        
        return max(0, synergy)  # Ensure non-negative synergy
    
    def simulate_alliance_selection(self, num_alliances: int = 8) -> Dict:
        """Simulate the VEX V5 alliance selection process."""
        rankings = self.get_team_rankings()
        top_teams = rankings.head(num_alliances * 2)['teamNumber'].tolist()
        
        alliances = []
        available_teams = top_teams.copy()
        
        for i in range(num_alliances):
            if len(available_teams) < 2:
                break
                
            # Captain is the highest ranked available team
            captain = available_teams.pop(0)
            
            # Find best partner (highest synergy with captain)
            captain_data = self.df[self.df['teamNumber'] == captain].iloc[0]
            best_partner = None
            best_synergy = -1
            
            for team_num in available_teams:
                team_data = self.df[self.df['teamNumber'] == team_num].iloc[0]
                synergy = self.calculate_alliance_synergy([captain_data, team_data])
                
                if synergy > best_synergy:
                    best_synergy = synergy
                    best_partner = team_num
            
            if best_partner:
                available_teams.remove(best_partner)
                alliances.append([captain, best_partner])
        
        return {
            "alliances": alliances,
            "available_teams": available_teams
        }
    
    def simulate_elimination_bracket(self, alliances: List[List[str]]) -> Dict:
        """Simulate a 2v2 elimination bracket tournament."""
        if len(alliances) < 2:
            return {"error": "Need at least 2 alliances"}
        
        # Ensure we have a power of 2 number of alliances
        while len(alliances) & (len(alliances) - 1) != 0:
            alliances.append(["BYE", "BYE"])
        
        bracket_results = []
        current_alliances = alliances.copy()
        
        round_num = 1
        while len(current_alliances) > 1:
            round_matches = []
            next_round_alliances = []
            
            for i in range(0, len(current_alliances), 2):
                if i + 1 < len(current_alliances):
                    alliance1 = current_alliances[i]
                    alliance2 = current_alliances[i + 1]
                    
                    if alliance1 == ["BYE", "BYE"]:
                        winner = alliance2
                    elif alliance2 == ["BYE", "BYE"]:
                        winner = alliance1
                    else:
                        prediction = self.predict_match_outcome(alliance1, alliance2)
                        winner = prediction["predicted_winner"]
                    
                    round_matches.append({
                        "alliance1": alliance1,
                        "alliance2": alliance2,
                        "winner": winner
                    })
                    next_round_alliances.append(winner)
            
            bracket_results.append({
                "round": round_num,
                "matches": round_matches
            })
            current_alliances = next_round_alliances
            round_num += 1
        
        return {
            "bracket_results": bracket_results,
            "final_winner": current_alliances[0] if current_alliances else None
        }
    
    def find_best_alliance_partners(self, captain_team: str, num_partners: int = 3) -> List[Dict]:
        """Find the best alliance partners for a given captain team."""
        captain_data = self.df[self.df['teamNumber'] == captain_team]
        if captain_data.empty:
            return []
        
        captain_stats = captain_data.iloc[0]
        
        # Calculate synergy scores with other teams
        synergy_scores = []
        
        for _, team in self.df.iterrows():
            if team['teamNumber'] != captain_team:
                # Calculate synergy based on complementary skills
                synergy = (
                    captain_stats['opr'] * team['opr'] * 0.3 +  # Both teams can score
                    (100 - captain_stats['dpr']) * (100 - team['dpr']) * 0.2 +  # Both teams defend well
                    captain_stats['apPerMatch'] * team['apPerMatch'] * 0.2 +  # Both teams autonomous
                    captain_stats['awpPerMatch'] * team['awpPerMatch'] * 0.3   # Both teams win points
                )
                
                synergy_scores.append({
                    "team_number": team['teamNumber'],
                    "team_name": team['teamName'],
                    "synergy_score": synergy,
                    "opr": team['opr'],
                    "dpr": team['dpr'],
                    "ap_per_match": team['apPerMatch'],
                    "awp_per_match": team['awpPerMatch']
                })
        
        # Sort by synergy score and return top partners
        synergy_scores.sort(key=lambda x: x['synergy_score'], reverse=True)
        return synergy_scores[:num_partners]

def main():
    st.markdown('<h1 class="main-header">🤖 VEX Tournament Analyzer</h1>', unsafe_allow_html=True)
    
    # Initialize the analyzer
    analyzer = VEXTournamentAnalyzer('GreatPlanesTeamData.xlsx')
    
    if analyzer.df is None:
        st.error("Failed to load data. Please check the Excel file.")
        return
    
    # Sidebar navigation
    st.sidebar.title("VEX V5 Analyzer")
    st.sidebar.markdown("*High School Tournament Analysis*")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Team Rankings", "2v2 Match Predictor", "Alliance Selection & Elimination", "Alliance Partner Finder", "Tournament Simulator"]
    )
    
    if page == "Team Rankings":
        show_team_rankings(analyzer)
    elif page == "2v2 Match Predictor":
        show_match_predictor(analyzer)
    elif page == "Alliance Selection & Elimination":
        show_elimination_predictor(analyzer)
    elif page == "Alliance Partner Finder":
        show_alliance_finder(analyzer)
    elif page == "Tournament Simulator":
        show_tournament_simulator(analyzer)

def show_team_rankings(analyzer):
    st.header("🏆 Team Rankings")
    
    rankings = analyzer.get_team_rankings()
    
    if rankings.empty:
        st.error("No rankings data available.")
        return
    
    # Display all teams
    st.subheader("All Teams (Cards)")
    for idx, team in rankings.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 2])
            with col1:
                st.metric("Rank", f"#{int(team['custom_ranking'])}")
            with col2:
                st.write(f"**{team['teamName']}** ({team['teamNumber']})")
            with col3:
                st.metric("Score", f"{team['composite_score']:.1f}")
    
    # Interactive ranking chart
    st.subheader("Ranking Visualization")
    
    # Create a scatter plot of teams
    fig = px.scatter(
        rankings,
        x='trueskill',
        y='ccwm',
        size='composite_score',
        color='totalWinningPercent',
        hover_data=['teamName', 'teamNumber', 'opr', 'dpr'],
        title="Team Performance Scatter Plot (All Teams)",
        labels={
            'trueskill': 'TrueSkill Rating',
            'ccwm': 'CCWM Score',
            'composite_score': 'Composite Score',
            'totalWinningPercent': 'Win Percentage (%)'
        }
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Full rankings table
    st.subheader("All Teams (Table)")
    st.dataframe(rankings, use_container_width=True)

def show_match_predictor(analyzer):
    st.header("🎯 2v2 Match Predictor")
    st.markdown("*Predict outcomes of VEX V5 High School 2v2 matches*")
    
    # Alliance selection
    st.subheader("Alliance Selection")
    team_options = analyzer.df['teamNumber'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Alliance 1")
        team1_1 = st.selectbox("Team 1:", team_options, key="alliance1_team1")
        team1_2_options = [t for t in team_options if t != team1_1]
        team1_2 = st.selectbox("Team 2:", team1_2_options, key="alliance1_team2")
    
    with col2:
        st.markdown("### Alliance 2")
        team2_1_options = [t for t in team_options if t not in [team1_1, team1_2]]
        team2_1 = st.selectbox("Team 1:", team2_1_options, key="alliance2_team1")
        team2_2_options = [t for t in team_options if t not in [team1_1, team1_2, team2_1]]
        team2_2 = st.selectbox("Team 2:", team2_2_options, key="alliance2_team2")
    
    if st.button("Predict 2v2 Match Outcome"):
        alliance1 = [team1_1, team1_2]
        alliance2 = [team2_1, team2_2]
        
        prediction = analyzer.predict_match_outcome(alliance1, alliance2)
        
        if "error" in prediction:
            st.error(prediction["error"])
        else:
            st.subheader("2v2 Match Prediction Results")
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown("### Alliance 1")
                for i, (team_num, team_name) in enumerate(zip(prediction['alliance1']['teams'], prediction['alliance1']['team_names'])):
                    st.write(f"**{team_name}** ({team_num})")
                
                st.metric("Combined Strength", f"{prediction['alliance1']['strength']:.1f}")
                st.metric("Synergy Bonus", f"{prediction['alliance1']['synergy']:.1f}")
                st.metric("Total Strength", f"{prediction['alliance1']['total_strength']:.1f}")
                
                win_prob = prediction['alliance1']['win_probability']
                if win_prob > 0.6:
                    st.markdown(f'<span class="prediction-high">Win Probability: {win_prob:.1%}</span>', unsafe_allow_html=True)
                elif win_prob > 0.4:
                    st.markdown(f'<span class="prediction-medium">Win Probability: {win_prob:.1%}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="prediction-low">Win Probability: {win_prob:.1%}</span>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### VS")
                st.markdown("<br><br><br><br>", unsafe_allow_html=True)
                predicted_winner = prediction['predicted_winner']
                if predicted_winner == alliance1:
                    st.markdown('<span class="prediction-high">🏆 Alliance 1 Wins</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="prediction-high">🏆 Alliance 2 Wins</span>', unsafe_allow_html=True)
            
            with col3:
                st.markdown("### Alliance 2")
                for i, (team_num, team_name) in enumerate(zip(prediction['alliance2']['teams'], prediction['alliance2']['team_names'])):
                    st.write(f"**{team_name}** ({team_num})")
                
                st.metric("Combined Strength", f"{prediction['alliance2']['strength']:.1f}")
                st.metric("Synergy Bonus", f"{prediction['alliance2']['synergy']:.1f}")
                st.metric("Total Strength", f"{prediction['alliance2']['total_strength']:.1f}")
                
                win_prob = prediction['alliance2']['win_probability']
                if win_prob > 0.6:
                    st.markdown(f'<span class="prediction-high">Win Probability: {win_prob:.1%}</span>', unsafe_allow_html=True)
                elif win_prob > 0.4:
                    st.markdown(f'<span class="prediction-medium">Win Probability: {win_prob:.1%}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="prediction-low">Win Probability: {win_prob:.1%}</span>', unsafe_allow_html=True)

def show_elimination_predictor(analyzer):
    st.header("🏁 VEX V5 Elimination Bracket Predictor")
    st.markdown("*Simulate alliance selection and elimination bracket for VEX V5 High School*")
    
    # Initialize session state
    if 'alliance_selection_results' not in st.session_state:
        st.session_state.alliance_selection_results = None
    
    # Alliance selection simulation
    st.subheader("Alliance Selection Simulation")
    
    num_alliances = st.slider("Number of alliances:", 2, 16, 8)
    
    if st.button("Simulate Alliance Selection"):
        selection = analyzer.simulate_alliance_selection(num_alliances)
        st.session_state.alliance_selection_results = selection
        
        st.subheader("Alliance Selection Results")
        
        for i, alliance in enumerate(selection["alliances"], 1):
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                st.metric("Alliance", f"#{i}")
            
            with col2:
                captain_data = analyzer.df[analyzer.df['teamNumber'] == alliance[0]].iloc[0]
                partner_data = analyzer.df[analyzer.df['teamNumber'] == alliance[1]].iloc[0]
                
                st.write(f"**Captain:** {captain_data['teamName']} ({alliance[0]})")
                st.write(f"**Partner:** {partner_data['teamName']} ({alliance[1]})")
            
            with col3:
                synergy = analyzer.calculate_alliance_synergy([captain_data, partner_data])
                st.metric("Synergy", f"{synergy:.1f}")
            
            st.markdown("---")
        
        st.success("✅ Alliance selection completed! You can now simulate the elimination bracket.")
    
    # Show elimination bracket simulation section
    st.subheader("Elimination Bracket Simulation")
    
    if st.session_state.alliance_selection_results is not None:
        st.info("💡 Alliance selection results are available. Click the button below to simulate the elimination bracket.")
        
        if st.button("Simulate Elimination Bracket"):
            bracket_simulation = analyzer.simulate_elimination_bracket(st.session_state.alliance_selection_results["alliances"])
            
            if "error" in bracket_simulation:
                st.error(bracket_simulation["error"])
            else:
                st.subheader("Elimination Bracket Results")
                
                for round_data in bracket_simulation["bracket_results"]:
                    st.markdown(f"### Round {round_data['round']}")
                    
                    for match in round_data["matches"]:
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            alliance1_names = []
                            for team_num in match['alliance1']:
                                team_data = analyzer.df[analyzer.df['teamNumber'] == team_num]
                                if not team_data.empty:
                                    alliance1_names.append(f"{team_data.iloc[0]['teamName']} ({team_num})")
                            st.write("**Alliance 1:**")
                            for name in alliance1_names:
                                st.write(f"• {name}")
                        
                        with col2:
                            if match['winner'] == match['alliance1']:
                                st.markdown('<span class="prediction-high">🏆</span>', unsafe_allow_html=True)
                            else:
                                st.write("vs")
                        
                        with col3:
                            alliance2_names = []
                            for team_num in match['alliance2']:
                                team_data = analyzer.df[analyzer.df['teamNumber'] == team_num]
                                if not team_data.empty:
                                    alliance2_names.append(f"{team_data.iloc[0]['teamName']} ({team_num})")
                            st.write("**Alliance 2:**")
                            for name in alliance2_names:
                                st.write(f"• {name}")
                            
                            if match['winner'] == match['alliance2']:
                                st.markdown('<span class="prediction-high">🏆</span>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                
                if bracket_simulation['final_winner']:
                    winner_names = []
                    for team_num in bracket_simulation['final_winner']:
                        team_data = analyzer.df[analyzer.df['teamNumber'] == team_num]
                        if not team_data.empty:
                            winner_names.append(f"{team_data.iloc[0]['teamName']} ({team_num})")
                    
                    st.success(f"🏆 **Tournament Champions:** {', '.join(winner_names)}**")
    else:
        st.warning("⚠️ Please run alliance selection first before simulating the elimination bracket.")

def show_alliance_finder(analyzer):
    st.header("🤝 Alliance Partner Finder")
    
    # Captain team selection
    captain_options = analyzer.df['teamNumber'].tolist()
    captain_team = st.selectbox("Select Captain Team:", captain_options)
    
    num_partners = st.slider("Number of recommended partners:", 1, 10, 3)
    
    if st.button("Find Best Alliance Partners"):
        partners = analyzer.find_best_alliance_partners(captain_team, num_partners)
        
        if partners:
            st.subheader(f"Best Alliance Partners for {captain_team}")
            
            for i, partner in enumerate(partners, 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                    
                    with col1:
                        st.metric("Rank", f"#{i}")
                    
                    with col2:
                        st.write(f"**{partner['team_name']}** ({partner['team_number']})")
                    
                    with col3:
                        st.metric("Synergy Score", f"{partner['synergy_score']:.1f}")
                    
                    with col4:
                        st.write(f"OPR: {partner['opr']:.1f}")
                        st.write(f"DPR: {partner['dpr']:.1f}")
                    
                    st.markdown("---")
        else:
            st.error("No alliance partners found.")

def show_tournament_simulator(analyzer):
    st.header("🎮 VEX V5 Tournament Simulator")
    st.markdown("*Simulate multiple tournament scenarios with alliance selection*")
    
    st.subheader("Simulation Parameters")
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        num_simulations = st.slider("Number of simulations:", 10, 1000, 100)
        num_alliances = st.selectbox("Number of alliances:", [4, 6, 8, 12, 16])
    
    with col2:
        include_alliance_selection = st.checkbox("Include alliance selection simulation", value=True)
        show_alliance_stats = st.checkbox("Show alliance statistics", value=True)
    
    if st.button("Run Tournament Simulation"):
        with st.spinner("Running simulations..."):
            winners = []
            alliance_stats = []
            
            for sim_num in range(num_simulations):
                # Simulate alliance selection
                if include_alliance_selection:
                    selection = analyzer.simulate_alliance_selection(num_alliances)
                    alliances = selection["alliances"]
                else:
                    # Use top teams directly
                    rankings = analyzer.get_team_rankings()
                    top_teams = rankings.head(num_alliances * 2)['teamNumber'].tolist()
                    alliances = []
                    for i in range(0, len(top_teams), 2):
                        if i + 1 < len(top_teams):
                            alliances.append([top_teams[i], top_teams[i + 1]])
                
                # Simulate elimination bracket
                bracket_simulation = analyzer.simulate_elimination_bracket(alliances)
                
                if bracket_simulation.get('final_winner'):
                    winners.append(bracket_simulation['final_winner'])
                    
                    if show_alliance_stats:
                        # Calculate alliance statistics
                        for alliance in alliances:
                            captain_data = analyzer.df[analyzer.df['teamNumber'] == alliance[0]].iloc[0]
                            partner_data = analyzer.df[analyzer.df['teamNumber'] == alliance[1]].iloc[0]
                            synergy = analyzer.calculate_alliance_synergy([captain_data, partner_data])
                            
                            alliance_stats.append({
                                'alliance': alliance,
                                'captain': alliance[0],
                                'partner': alliance[1],
                                'synergy': synergy,
                                'won_tournament': bracket_simulation['final_winner'] == alliance
                            })
            
            # Analyze results
            st.subheader("Simulation Results")
            st.write(f"Ran {num_simulations} simulations with {num_alliances} alliances")
            
            # Count winning alliances
            winner_alliances = []
            for winner in winners:
                winner_alliances.append(tuple(sorted(winner)))
            
            winner_counts = pd.Series(winner_alliances).value_counts()
            
            # Display top winning alliances
            st.subheader("Most Successful Alliances")
            for i, (alliance_tuple, count) in enumerate(winner_counts.head(10).items(), 1):
                percentage = (count / num_simulations) * 100
                alliance_list = list(alliance_tuple)
                
                col1, col2, col3 = st.columns([1, 3, 2])
                with col1:
                    st.metric("Rank", f"#{i}")
                with col2:
                    alliance_names = []
                    for team_num in alliance_list:
                        team_data = analyzer.df[analyzer.df['teamNumber'] == team_num]
                        if not team_data.empty:
                            alliance_names.append(f"{team_data.iloc[0]['teamName']} ({team_num})")
                    st.write("**Alliance:**")
                    for name in alliance_names:
                        st.write(f"• {name}")
                with col3:
                    st.metric("Win Rate", f"{percentage:.1f}%")
                
                st.markdown("---")
            
            # Create visualization
            if len(winner_counts) > 0:
                fig = px.bar(
                    x=[f"Alliance {i+1}" for i in range(len(winner_counts.head(10)))],
                    y=winner_counts.head(10).values,
                    title="Tournament Win Frequency by Alliance",
                    labels={'x': 'Alliance', 'y': 'Number of Wins'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show alliance statistics if requested
            if show_alliance_stats and alliance_stats:
                st.subheader("Alliance Performance Statistics")
                
                df_stats = pd.DataFrame(alliance_stats)
                
                # Calculate average synergy for winning vs losing alliances
                winning_synergy = df_stats[df_stats['won_tournament']]['synergy'].mean()
                losing_synergy = df_stats[~df_stats['won_tournament']]['synergy'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Synergy (Winning Alliances)", f"{winning_synergy:.1f}")
                with col2:
                    st.metric("Avg Synergy (Losing Alliances)", f"{losing_synergy:.1f}")
                
                # Show synergy distribution
                fig_synergy = px.histogram(
                    df_stats,
                    x='synergy',
                    color='won_tournament',
                    title="Synergy Distribution by Tournament Outcome",
                    labels={'synergy': 'Alliance Synergy Score', 'count': 'Number of Alliances'}
                )
                st.plotly_chart(fig_synergy, use_container_width=True)

if __name__ == "__main__":
    main()
