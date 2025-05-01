import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Gaming Engagement Predictor", layout="wide")

st.markdown(
    """
    <style>
    body {background-color: #0d47a1; color: white;}
    .stApp {background-color: #0d47a1; color: white;}
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label, .st-bb, .st-cb {color: white !important;}
    .stButton>button {background-color: white !important; color: #0d47a1 !important; font-weight: bold;}
    .welcome-text {color: #ffeb3b; font-weight: bold; font-size: 18px;}
    </style>
    """,
    unsafe_allow_html=True
)

USER_CREDENTIALS = {"Arun": "Loginpage@123"}

def login():
    st.title("🔐 Login to Access the App")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"✅ Welcome, {username}!")
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.success("You have been logged out")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    st.title("🎮 Engagement Level Predictor")

    if st.button("Logout"):
        logout()

    model = joblib.load("engagement_model.pkl")
    scaler = joblib.load("scaler.pkl")

    option = st.radio("Choose Prediction Method", ["🔘 Manual Input", "📂 Upload CSV for Bulk Prediction"])

    if option == "🔘 Manual Input":
        age = st.number_input("Age", min_value=15, max_value=49, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Player Location", ["Europe", "Other", "USA"])
        game_genre = st.selectbox("Favorite Game Genre", ["Action", "Adventure", "Puzzle", "RPG", "Simulation", "Sports", "Strategy", "Other"])
        play_time = st.number_input("Average Play Time (Hours)", min_value=0.0, step=0.5)
        in_game_purchases = st.radio("In-Game Purchases", ["No", "Yes"])
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
            age, gender_encoded, play_time, 1 if in_game_purchases == "Yes" else 0,
            game_difficulty_encoded, sessions_per_week, avg_session_duration, player_level, achievements,
            location_europe, location_other, location_usa,
            genre_rpg, genre_simulation, genre_sports, genre_strategy
        ]])

        if st.button("Predict Engagement Level"):
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            prediction_value = int(prediction.item())
            engagement_labels = {1: "Low", 2: "Medium", 3: "High"}
            st.success(f"🎯 Predicted Engagement Level: **{engagement_labels.get(prediction_value, 'Unknown')}**")

    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            try:
                # Encode gender
                df["gender_encoded"] = df["Gender"].map({"Male": 0, "Female": 1})
                df["in_game_purchases"] = df["In_Game_Purchases"].map({"No": 0, "Yes": 1})
                df["game_difficulty_encoded"] = df["Game_Difficulty"].map({'Easy': 1, 'Medium': 2, 'Hard': 3})

                # Location
                df["location_europe"] = df["Location"].apply(lambda x: 1 if x == "Europe" else 0)
                df["location_other"] = df["Location"].apply(lambda x: 1 if x == "Other" else 0)
                df["location_usa"] = df["Location"].apply(lambda x: 1 if x == "USA" else 0)

                # Game Genre
                df["genre_rpg"] = df["Game_Genre"].apply(lambda x: 1 if x == "RPG" else 0)
                df["genre_simulation"] = df["Game_Genre"].apply(lambda x: 1 if x == "Simulation" else 0)
                df["genre_sports"] = df["Game_Genre"].apply(lambda x: 1 if x == "Sports" else 0)
                df["genre_strategy"] = df["Game_Genre"].apply(lambda x: 1 if x == "Strategy" else 0)

                feature_cols = [
                    "Age", "gender_encoded", "Play_Time", "in_game_purchases", "game_difficulty_encoded",
                    "Sessions_Per_Week", "Avg_Session_Duration", "Player_Level", "Achievements",
                    "location_europe", "location_other", "location_usa",
                    "genre_rpg", "genre_simulation", "genre_sports", "genre_strategy"
                ]

                input_data = df[feature_cols]
                scaled_input = scaler.transform(input_data)
                predictions = model.predict(scaled_input)
                df["Engagement_Level"] = predictions
                engagement_map = {1: "Low", 2: "Medium", 3: "High"}
                df["Engagement_Level_Label"] = df["Engagement_Level"].map(engagement_map)

                st.success("✅ Predictions completed. Here's a preview:")
                st.dataframe(df[["Age", "Gender", "Location", "Game_Genre", "Engagement_Level_Label"]])

                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions as CSV", csv_output, file_name="engagement_predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
