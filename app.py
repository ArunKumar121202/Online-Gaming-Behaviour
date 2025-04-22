import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("engagement_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🎮 Online Gaming Engagement Level Predictor")
st.write("Fill out the player's details to predict their engagement level:")

# Input: Age within dataset range
age = st.number_input("Age", min_value=15, max_value=49, step=1) 

# Input: Gender
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# Input: Favorite Game Genre
game_genre = st.selectbox("Favorite Game Genre", [
    "Action", "Adventure", "Puzzle", "RPG", "Simulation", "Sports", "Strategy", "Other"
])

# Input: Average Play Time
play_time = st.number_input("Average Play Time (Hours)", min_value=0.0, step=0.5)

# ✅ Binary input for In-Game Purchases
in_game_purchases = st.radio("In-Game Purchases", ["No", "Yes"])
in_game_purchases = 1 if in_game_purchases == "Yes" else 0

# Input: Game Difficulty
game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])

# Other inputs
sessions_per_week = st.number_input("Sessions per Week", min_value=0, step=1)
avg_session_duration = st.number_input("Avg. Session Duration (minutes)", min_value=0, step=1)
player_level = st.number_input("Player Level", min_value=0, step=1)
achievements = st.number_input("Achievements Unlocked", min_value=0, step=1)

# Encoding categorical variables
gender_map = {"Male": 0, "Female": 1, "Other": 2}
genre_map = {
    'Action': 0, 'Adventure': 1, 'Puzzle': 2, 'RPG': 3,
    'Simulation': 4, 'Sports': 5, 'Strategy': 6, 'Other': 7
}
difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}

# Create input array
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

# Predict button
if st.button("Predict Engagement Level"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"🎯 Predicted Engagement Level: **{prediction}**")
