# 🤖 VEX V5 Tournament Analyzer

A comprehensive web application for analyzing VEX V5 High School tournament data, predicting 2v2 match outcomes, and simulating alliance selection and elimination brackets.

## Features

### 🏆 Team Rankings
- **Composite Scoring System**: Analyzes multiple statistics including TrueSkill, CCWM, win percentages, OPR, DPR, and more
- **Interactive Visualizations**: Scatter plots showing team performance across different metrics
- **Complete Rankings Table**: View all teams with detailed statistics

### 🎯 2v2 Match Predictor
- **Alliance vs Alliance Predictions**: Predict outcomes of 2v2 matches between alliances
- **Synergy Analysis**: Calculate alliance synergy bonuses based on complementary skills
- **Win Probability Calculations**: Uses combined team strengths and synergy to determine win likelihood
- **Detailed Match Analysis**: Shows individual team strengths, synergy bonuses, and predicted winners

### 🏁 Alliance Selection & Elimination Predictor
- **Realistic Alliance Selection**: Simulate the VEX V5 alliance selection process with captains and partners
- **Synergy-Based Partner Selection**: Captains choose partners based on synergy scores
- **2v2 Elimination Brackets**: Simulate elimination tournaments with alliance-based matches
- **Tournament Winner Prediction**: Identify the most likely alliance to win the tournament

### 🤝 Alliance Partner Finder
- **Synergy Analysis**: Find the best alliance partners for any captain team
- **Complementary Skills**: Considers OPR, DPR, autonomous, and win point capabilities
- **Ranked Recommendations**: Get top alliance partner suggestions

### 🎮 Tournament Simulator
- **Monte Carlo Simulation**: Run hundreds of tournament simulations with alliance selection
- **Alliance Performance Analysis**: See which alliances win most frequently
- **Synergy Impact Analysis**: Analyze how alliance synergy affects tournament outcomes
- **Flexible Parameters**: Choose number of alliances, simulation count, and analysis options

## Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Place your Excel file** (`GreatPlanesTeamData.xlsx`) in the same directory as `app.py`
2. **Run the application**:
   ```bash
   streamlit run app.py
   ```
3. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`)

## Data Format

The application expects an Excel file with the following columns:
- `teamNumber`: Team identifier
- `teamName`: Team name
- `trueskill`: TrueSkill rating
- `ccwm`: CCWM score
- `totalWinningPercent`: Overall win percentage
- `eliminationWinningPercent`: Elimination round win percentage
- `opr`: Offensive Power Rating
- `dpr`: Defensive Power Rating
- `apPerMatch`: Autonomous points per match
- `awpPerMatch`: Alliance win points per match
- And other VEX tournament statistics

## Scoring Algorithm

The composite scoring system uses weighted factors:
- **TrueSkill (25%)**: Overall team skill rating
- **CCWM (20%)**: Contribution to winning margin
- **Total Win % (15%)**: Overall match performance
- **Elimination Win % (15%)**: Performance in elimination rounds
- **OPR (10%)**: Offensive capability
- **DPR (5%)**: Defensive capability (inverted - lower is better)
- **AP per Match (5%)**: Autonomous performance
- **AWP per Match (5%)**: Alliance win point contribution

## Features Overview

### Navigation
Use the sidebar to switch between different analysis tools:
- **Team Rankings**: View comprehensive team rankings
- **Match Predictor**: Predict individual match outcomes
- **Elimination Predictor**: Simulate elimination brackets
- **Alliance Partner Finder**: Find optimal alliance partners
- **Tournament Simulator**: Run statistical tournament simulations

### Interactive Elements
- **Team Selection**: Dropdown menus for easy team selection
- **Parameter Adjustment**: Sliders and checkboxes for customization
- **Real-time Updates**: Results update immediately when parameters change
- **Visualizations**: Interactive charts and graphs for data analysis

## Technical Details

- **Framework**: Streamlit for web interface
- **Data Processing**: Pandas for data manipulation
- **Visualizations**: Plotly for interactive charts
- **Machine Learning**: Scikit-learn for predictive models
- **File Format**: Excel (.xlsx) support via openpyxl

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the analyzer.

## License

This project is open source and available under the MIT License.
