import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Gaming Engagement Predictor & Analysis", layout="wide")

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

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

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
        uploaded_file = st.file_uploader("Upload your dataset CSV", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            st.subheader("Basic Statistics")
            st.dataframe(df.describe())

            st.subheader("Missing Values")
            st.dataframe(df.isnull().sum())

            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            if 'Engagement_Level' in df.columns:
                st.subheader("Distribution of Engagement Levels")
                fig2, ax2 = plt.subplots()
                sns.countplot(data=df, x='Engagement_Level', palette='viridis', ax=ax2)
                ax2.set_xlabel("Engagement Level")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)

            if 'Age' in df.columns:
                st.subheader("Age Distribution")
                fig3, ax3 = plt.subplots()
                sns.histplot(df['Age'], bins=20, kde=True, color='orange', ax=ax3)
                ax3.set_xlabel("Age")
                ax3.set_title("Age Distribution")
                st.pyplot(fig3)

            if 'Gender' in df.columns:
                st.subheader("Gender Distribution")
                fig4, ax4 = plt.subplots()
                sns.countplot(data=df, x='Gender', palette='Set2', ax=ax4)
                st.pyplot(fig4)

            if 'Game_Genre' in df.columns:
                st.subheader("Preferred Game Genre")
                fig5, ax5 = plt.subplots()
                df['Game_Genre'].value_counts().plot(kind='bar', color='purple', ax=ax5)
                ax5.set_ylabel("Count")
                st.pyplot(fig5)

            if 'Location' in df.columns:
                st.subheader("Player Location")
                fig6, ax6 = plt.subplots()
                df['Location'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax6)
                ax6.set_ylabel("")
                st.pyplot(fig6)

        else:
            st.warning("Please upload a CSV file to continue.")
