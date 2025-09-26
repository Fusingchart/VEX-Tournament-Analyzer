#!/usr/bin/env python3
"""
VEX Tournament Analyzer Demo
Demonstrates the key features of the analyzer
"""

from app import VEXTournamentAnalyzer
import pandas as pd

def main():
    print("🤖 VEX Tournament Analyzer Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = VEXTournamentAnalyzer('GreatPlanesTeamData.xlsx')
    
    if analyzer.df is None:
        print("❌ Failed to load data")
        return
    
    print(f"📊 Loaded data for {len(analyzer.df)} teams")
    print()
    
    # Show top 5 rankings
    print("🏆 TOP 5 TEAM RANKINGS")
    print("-" * 30)
    rankings = analyzer.get_team_rankings()
    top_5 = rankings.head(5)
    
    for _, team in top_5.iterrows():
        print(f"#{int(team['custom_ranking'])} {team['teamName']} ({team['teamNumber']})")
        print(f"   Composite Score: {team['composite_score']:.1f}")
        print(f"   TrueSkill: {team['trueskill']:.1f}, CCWM: {team['ccwm']:.1f}")
        print(f"   Win %: {team['totalWinningPercent']:.1f}%, OPR: {team['opr']:.1f}")
        print()
    
    # Show 2v2 match prediction example
    print("🎯 2V2 MATCH PREDICTION EXAMPLE")
    print("-" * 30)
    if len(rankings) >= 4:
        alliance1 = [rankings.iloc[0]['teamNumber'], rankings.iloc[1]['teamNumber']]
        alliance2 = [rankings.iloc[2]['teamNumber'], rankings.iloc[3]['teamNumber']]
        
        prediction = analyzer.predict_match_outcome(alliance1, alliance2)
        
        print(f"Alliance 1: {alliance1[0]} + {alliance1[1]}")
        print(f"Alliance 2: {alliance2[0]} + {alliance2[1]}")
        print(f"Alliance 1 Win Probability: {prediction['alliance1']['win_probability']:.1%}")
        print(f"Alliance 2 Win Probability: {prediction['alliance2']['win_probability']:.1%}")
        print(f"Alliance 1 Synergy: {prediction['alliance1']['synergy']:.1f}")
        print(f"Alliance 2 Synergy: {prediction['alliance2']['synergy']:.1f}")
        print(f"Predicted Winner: Alliance {1 if prediction['predicted_winner'] == alliance1 else 2}")
        print()
    
    # Show alliance partner example
    print("🤝 ALLIANCE PARTNER EXAMPLE")
    print("-" * 30)
    if len(rankings) >= 1:
        captain = rankings.iloc[0]['teamNumber']
        partners = analyzer.find_best_alliance_partners(captain, 3)
        
        print(f"Best alliance partners for {captain}:")
        for i, partner in enumerate(partners, 1):
            print(f"{i}. {partner['team_name']} ({partner['team_number']})")
            print(f"   Synergy Score: {partner['synergy_score']:.1f}")
            print(f"   OPR: {partner['opr']:.1f}, DPR: {partner['dpr']:.1f}")
        print()
    
    # Show alliance selection and elimination simulation example
    print("🏁 ALLIANCE SELECTION & ELIMINATION EXAMPLE")
    print("-" * 30)
    if len(rankings) >= 8:
        # Simulate alliance selection
        selection = analyzer.simulate_alliance_selection(4)
        
        print("Alliance Selection Results:")
        for i, alliance in enumerate(selection['alliances'], 1):
            captain_data = analyzer.df[analyzer.df['teamNumber'] == alliance[0]].iloc[0]
            partner_data = analyzer.df[analyzer.df['teamNumber'] == alliance[1]].iloc[0]
            synergy = analyzer.calculate_alliance_synergy([captain_data, partner_data])
            
            print(f"  Alliance {i}: {alliance[0]} (Captain) + {alliance[1]} (Partner)")
            print(f"    Synergy Score: {synergy:.1f}")
        print()
        
        # Simulate elimination bracket
        bracket_simulation = analyzer.simulate_elimination_bracket(selection['alliances'])
        
        print("Elimination Bracket Results:")
        for round_data in bracket_simulation['bracket_results']:
            print(f"Round {round_data['round']}:")
            for match in round_data['matches']:
                alliance1_str = f"{match['alliance1'][0]}+{match['alliance1'][1]}"
                alliance2_str = f"{match['alliance2'][0]}+{match['alliance2'][1]}"
                winner_str = f"{match['winner'][0]}+{match['winner'][1]}"
                print(f"  {alliance1_str} vs {alliance2_str} → Winner: {winner_str}")
            print()
        
        if bracket_simulation['final_winner']:
            winner_str = f"{bracket_simulation['final_winner'][0]}+{bracket_simulation['final_winner'][1]}"
            print(f"🏆 Tournament Champions: {winner_str}")
        print()
    
    print("✅ Demo completed! Run 'streamlit run app.py' to use the full application.")

if __name__ == "__main__":
    main()
