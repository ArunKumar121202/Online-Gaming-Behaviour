import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("engagement_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🎮 Online Gaming Engagement Level Predictor")
st.write("Provide the player data below to predict their engagement level:")

# Correct feature names from your dataset
feature_names = [
    'Age', 'Gender', 'Time_Spent_Playing', 'Frequency_of_Purchase',
    'Favorite_Game_Genre', 'Hours_Watched', 'In_Game_Purchases',
    'Communication_Tool_Usage', 'Skill_Level'
]

# Collect user inputs
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, step=1.0)
    user_inputs.append(value)

# Prediction
if st.button("Predict Engagement Level"):
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🎯 Predicted Engagement Level: **{prediction}**")
