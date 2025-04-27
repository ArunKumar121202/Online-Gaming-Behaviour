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
    menu = st.sidebar.radio("Choose Section", ["Predict Engagement", "Analyze Dataset"])

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
        
        # Load the dataset directly from a file
        df = pd.read_csv("online_gaming_behavior_dataset.csv")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Statistics")
        st.dataframe(df.describe())

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.corr(numeric_only=True)
        im = ax.imshow(corr, cmap="coolwarm")
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im)
        st.pyplot(fig)

        # Side-by-side charts with better alignment
        col1, col2 = st.columns([2, 2])  # Equal column width for both charts

        with col1:
            st.markdown("#### 🎮 Engagement Level Distribution")
            if 'EngagementLevel' in df.columns:
                fig, ax = plt.subplots()
                order = df['EngagementLevel'].value_counts().index
                sns.countplot(data=df, x='EngagementLevel', order=order, palette='Blues', ax=ax)
                for p in ax.patches:
                    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                                textcoords='offset points')
                ax.set_ylabel("Count")
                st.pyplot(fig)

        with col2:
            st.markdown("#### 👤 Gender Distribution")
            if 'Gender' in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='Gender', palette='Set2', ax=ax)
                for p in ax.patches:
                    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                                textcoords='offset points')
                ax.set_ylabel("Count")
                st.pyplot(fig)

        col3, col4 = st.columns([2, 2])  # Equal column width for both charts

        with col3:
            st.markdown("#### 🕹️ Favorite Game Genre")
            if 'GameGenre' in df.columns:
                genre_counts = df['GameGenre'].value_counts()
                fig, ax = plt.subplots()
                genre_counts.plot(kind='bar', color='mediumseagreen', edgecolor='black', ax=ax)
                for i, v in enumerate(genre_counts):
                    ax.text(i, v + 1, str(v), ha='center')
                ax.set_xlabel("Game Genre")
                ax.set_ylabel("Count")
                st.pyplot(fig)

        with col4:
            st.markdown("#### 🌍 Player Location")
            if 'Location' in df.columns:
                location_counts = df['Location'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
                ax.axis('equal')
                st.pyplot(fig)
