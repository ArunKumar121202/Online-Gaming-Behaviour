import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Streamlit page settings
st.set_page_config(page_title="Gaming Engagement Predictor & Analysis", layout="wide")

# Custom CSS
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

# Login credentials
USER_CREDENTIALS = {"Arun": "Loginpage@123"}

# Login function
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

# Check login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# App Flow
if not st.session_state["logged_in"]:
    login()
else:
    menu = st.sidebar.selectbox("Choose Section", ["Predict Engagement", "Analyze Dataset"])

    if menu == "Predict Engagement":
        model = joblib.load("engagement_model.pkl")
        scaler = joblib.load("scaler.pkl")

        st.title("🎮 Engagement Level Predictor")
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

    elif menu == "Analyze Dataset":
        st.title("📊 Online Gaming Behavior Dataset Analysis")

        st.sidebar.subheader("Choose Analysis Section")
        analysis_section = st.sidebar.radio(
            "Select Section",
            [
                "Player Demographics",
                "Gameplay Behavior",
                "Player Engagement",
                "Purchase Behavior",
                "Player Level & Progression",
                "Location-Based Insights"
            ]
        )

        if analysis_section == "Player Demographics":
            st.header("👤 Player Demographics Analysis")
            total_players = df.shape[0]
            avg_age = round(df['Age'].mean(), 2)
            gender_counts = df['Gender'].value_counts()
            gender_ratio = f"{gender_counts.get('Male',0)}M : {gender_counts.get('Female',0)}F"

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Players", total_players)
            col2.metric("Average Age", avg_age)
            col3.metric("Gender Ratio", gender_ratio)

            st.subheader("Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['Age'], kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)

            st.subheader("Gender Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='Gender', palette='Set2', ax=ax)
            st.pyplot(fig)

        elif analysis_section == "Gameplay Behavior":
            st.header("🎮 Gameplay Behavior Analysis")
            avg_play_time = round(df['Play_Time_Hours'].mean(), 2)
            avg_sessions = round(df['Sessions_per_Week'].mean(), 2)
            avg_duration = round(df['Avg_Session_Duration_mins'].mean(), 2)

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Play Time (Hours)", avg_play_time)
            col2.metric("Sessions/Week", avg_sessions)
            col3.metric("Avg Session Duration (mins)", avg_duration)

            st.subheader("Game Difficulty Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='Game_Difficulty', palette='pastel', ax=ax)
            st.pyplot(fig)

        elif analysis_section == "Player Engagement":
            st.header("🎯 Player Engagement Analysis")
            engagement_counts = df['EngagementLevel'].value_counts()
            avg_achievements = round(df['Achievements_Unlocked'].mean(), 2)

            col1, col2 = st.columns(2)
            col1.metric("Most Common Engagement", engagement_counts.idxmax())
            col2.metric("Avg Achievements Unlocked", avg_achievements)

            st.subheader("Engagement Level Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='EngagementLevel', order=sorted(df['EngagementLevel'].unique()), palette='Blues', ax=ax)
            st.pyplot(fig)

        elif analysis_section == "Purchase Behavior":
            st.header("🛒 Purchase Behavior Analysis")
            purchase_counts = df['InGamePurchases'].value_counts()
            col1, col2 = st.columns(2)
            col1.metric("Players with Purchases", purchase_counts.get(1, 0))
            col2.metric("Players without Purchases", purchase_counts.get(0, 0))
            st.subheader("In-Game Purchases Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='InGamePurchases', palette='cool', ax=ax)
            ax.set_xticklabels(['No Purchase', 'Purchase'])  # Label 0 and 1 meaningfully
            st.pyplot(fig)

        elif analysis_section == "Player Level & Progression":
            st.header("📈 Player Level & Progression Analysis")
            avg_player_level = round(df['PlayerLevel'].mean(), 2)

            col1, col2 = st.columns(2)
            col1.metric("Average Player Level", avg_player_level)
            col2.metric("Max Achievements", df['Achievements_Unlocked'].max())

            st.subheader("Player Level Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['PlayerLevel'], kde=True, color='orange', ax=ax)
            st.pyplot(fig)

        elif analysis_section == "Location-Based Insights":
            st.header("🌍 Location Based Insights")
            location_counts = df['Location'].value_counts()

            col1, col2 = st.columns(2)
            col1.metric("Top Location", location_counts.idxmax())
            col2.metric("Players from Top Location", location_counts.max())

            st.subheader("Location Distribution")
            fig, ax = plt.subplots()
            ax.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            ax.axis('equal')
            st.pyplot(fig)
