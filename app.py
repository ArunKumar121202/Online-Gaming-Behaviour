import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Streamlit page settings
st.set_page_config(page_title="Gaming Engagement Predictor & Insights", layout="wide")

# Custom CSS for professional clean look
st.markdown(
    """
    <style>
    body {background-color: #000000; color: white;}
    .stApp {background-color: #000000; color: white;}
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label, .stSlider label, .st-bb, .st-cb {color: white !important;}
    .stButton>button {
        background-color: #04AA6D !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #039f62 !important;
        color: #fff !important;
        transform: scale(1.05);
    }
    .welcome-text {
        color: #00FF00;
        font-weight: bold;
        font-size: 22px;
        margin-bottom: 20px;
    }
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
            st.error("❌ Invalid username or password")

# Check login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# App Flow
if not st.session_state["logged_in"]:
    login()
else:
    st.markdown(f"<div class='welcome-text'>Welcome, {st.session_state['username']}! 🚀</div>", unsafe_allow_html=True)
    
    menu = st.sidebar.selectbox("Choose Section", ["Predict Engagement", "Analyze Dataset"])

    if menu == "Predict Engagement":
        model = joblib.load("engagement_model.pkl")
        scaler = joblib.load("scaler.pkl")

        st.title("🎮 Engagement Level Predictor")
        with st.form("prediction_form"):
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
            submit = st.form_submit_button("Predict Engagement Level")

        if submit:
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

            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            prediction_value = int(prediction.item())
            engagement_labels = {1: "Low", 2: "Medium", 3: "High"}
            st.success(f"🎯 Predicted Engagement Level: **{engagement_labels.get(prediction_value, 'Unknown')}**")

    elif menu == "Analyze Dataset":
        st.title("📊 Online Gaming Behavior - Dataset Insights")
        
        # Top KPIs
        total_players = df.shape[0]
        avg_age = round(df['Age'].mean(), 2)
        male_count = df[df['Gender'] == 'Male'].shape[0]
        female_count = df[df['Gender'] == 'Female'].shape[0]
        purchase_rate = round(df['InGamePurchases'].mean() * 100, 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Players", total_players)
        col2.metric("Average Age", avg_age)
        col3.metric("Males", male_count)
        col4.metric("Females", female_count)

        st.divider()

        st.subheader("1️⃣ Player Demographics")
        fig, ax = plt.subplots(1,2, figsize=(16,6))
        sns.histplot(df['Age'], kde=True, color='cyan', ax=ax[0])
        ax[0].set_title("Age Distribution")
        sns.countplot(data=df, x='Gender', palette='pastel', ax=ax[1])
        ax[1].set_title("Gender Distribution")
        st.pyplot(fig)

        st.divider()

        st.subheader("2️⃣ Gameplay Behavior")
        fig, ax = plt.subplots(1,2, figsize=(16,6))
        sns.histplot(df['Play TimeHours'], kde=True, color='lightgreen', ax=ax[0])
        ax[0].set_title("Play Time Distribution")
        sns.countplot(data=df, x='Game_Difficulty', palette='muted', ax=ax[1])
        ax[1].set_title("Game Difficulty Levels")
        st.pyplot(fig)

        st.divider()

        st.subheader("3️⃣ Player Engagement")
        engagement_mapping = {1: "Low", 2: "Medium", 3: "High"}
        df['EngagementLevel_label'] = df['EngagementLevel'].map(engagement_mapping)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.countplot(data=df, x='EngagementLevel_label', palette='cool', order=["Low", "Medium", "High"], ax=ax)
        ax.set_title("Player Engagement Levels")
        st.pyplot(fig)

        st.divider()

        st.subheader("4️⃣ Purchase Behavior")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.countplot(data=df, x='InGamePurchases', palette='rocket', ax=ax)
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_title("In-Game Purchase Behavior")
        st.pyplot(fig)

        st.divider()

        st.subheader("5️⃣ Player Progression")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(df['PlayerLevel'], kde=True, color='gold', ax=ax)
        ax.set_title("Player Level Distribution")
        st.pyplot(fig)

        st.divider()

        st.subheader("6️⃣ Location Based Insights")
        fig, ax = plt.subplots(figsize=(8,8))
        location_counts = df['Location'].value_counts()
        ax.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
        ax.axis('equal')
        st.pyplot(fig)
