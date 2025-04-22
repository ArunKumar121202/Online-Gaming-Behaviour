import streamlit as st

# Dummy user credentials (you can replace or secure these later)
USER_CREDENTIALS = {
    "admin": "password123",
    "player1": "gaming2024",
    "testuser": "1234"
}

# Login function
def login():
    st.title("🔐 Login to Access the Predictor")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")

# Check login state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    # Your main predictor app goes here
    import numpy as np
    import joblib

    model = joblib.load("engagement_model.pkl")
    scaler = joblib.load("scaler.pkl")

    st.title("🎮 Online Gaming Engagement Level Predictor")
    st.write(f"Welcome, **{st.session_state['username']}**! Fill out the player's details to predict their engagement level:")

    age = st.number_input("Age", min_value=15, max_value=49, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Player Location", ["Europe", "Other", "USA"])
    game_genre = st.selectbox("Favorite Game Genre", [
        "Action", "Adventure", "Puzzle", "RPG", "Simulation", "Sports", "Strategy", "Other"
    ])
    play_time = st.number_input("Average Play Time (Hours)", min_value=0.0, step=0.5)
    in_game_purchases = st.radio("In-Game Purchases", ["No", "Yes"])
    in_game_purchases = 1 if in_game_purchases == "Yes" else 0
    game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
    sessions_per_week = st.number_input("Sessions per Week", min_value=0, step=1)
    avg_session_duration = st.number_input("Avg. Session Duration (minutes)", min_value=0, step=1)
    player_level = st.number_input("Player Level", min_value=0, step=1)
    achievements = st.number_input("Achievements Unlocked", min_value=0, step=1)

    gender_encoded = 0 if gender == "Male" else 1
    difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
    game_difficulty_encoded = difficulty_map[game_difficulty]
    location_europe = 1 if location == "Europe" else 0
    location_other = 1 if location == "Other" else 0
    location_usa = 1 if location == "USA" else 0
    genre_rpg = 1 if game_genre == "RPG" else 0
    genre_simulation = 1 if game_genre == "Simulation" else 0
    genre_sports = 1 if game_genre == "Sports" else 0
    genre_strategy = 1 if game_genre == "Strategy" else 0

    input_data = np.array([[
        age, gender_encoded, play_time, in_game_purchases, game_difficulty_encoded,
        sessions_per_week, avg_session_duration, player_level, achievements,
        location_europe, location_other, location_usa,
        genre_rpg, genre_simulation, genre_sports, genre_strategy
    ]])

    if st.button("Predict Engagement Level"):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        prediction_value = int(prediction.item())
        engagement_labels = {1: "Low", 2: "Medium", 3: "High"}
        st.success(f"🎯 Predicted Engagement Level: **{engagement_labels.get(prediction_value, 'Unknown')}**")
