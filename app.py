import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

cities = [
    "Chennai", "Mumbai", "Kolkata", "Bengaluru", "Hyderabad",
    "Jaipur", "Delhi", "Mohali", "Ahmedabad", "Lucknow"
]

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams))

selected_city = st.selectbox('City', sorted(cities))
target = st.number_input('Target Score', min_value=0, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Include match_id because the pipeline expects it
    input_df = pd.DataFrame({
        'match_id': [0],  # dummy value
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict
    result = pipe.predict_proba(input_df)
    st.header(f"{batting_team} Win Probability: {round(result[0][1]*100, 2)}%")
