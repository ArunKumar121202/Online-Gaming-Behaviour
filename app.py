import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("engagement_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🎮 Online Gaming Engagement Level Predictor")
st.write("Fill out the player's details to predict their engagement level:")
# Collect inputs
age_min = df['Age'].min()
age_max = df['Age'].max()
age = st.number_input("Age", min_value=int(age_min), max_value=int(age_max), step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
game_genre = st.selectbox("Favorite Game Genre", ["Action", "Adventure", "Puzzle", "RPG", "Simulation", "Sports", "Strategy", "Other"])
play_time = st.number_input("Average Play Time (Hours)", min_value=0.0, step=0.5)
in_game_purchases = st.number_input("In-Game Purchases", min_value=0, step=1)
game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
sessions_per_week = st.number_input("Sessions per Week", min_value=0, step=1)
avg_session_duration = st.number_input("Avg. Session Duration (minutes)", min_value=0, step=1)
player_level = st.number_input("Player Level", min_value=0, step=1)
achievements = st.number_input("Achievements Unlocked", min_value=0, step=1)

# Map categorical to numeric (ensure this matches your model encoding)
gender_map = {"Male": 0, "Female": 1, "Other": 2}
genre_map = {'Action': 0, 'Adventure': 1, 'Puzzle': 2, 'RPG': 3, 'Simulation': 4, 'Sports': 5, 'Strategy': 6, 'Other': 7}
difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}

# Transform inputs
input_data = np.array([
    age,
    gender_map[gender],
    genre_map[game_genre],
    play_time,
    in_game_purchases,
    difficulty_map[game_difficulty],
    sessions_per_week,
    avg_session_duration,
    player_level,
    achievements
]).reshape(1, -1)

# Prediction
if st.button("Predict Engagement Level"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"🎯 Predicted Engagement Level: **{prediction}**")
