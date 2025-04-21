import streamlit as st
import pandas as pd
import numpy as np

# Page title
st.set_page_config(page_title="Online Gaming Behavior", layout="wide")
st.title("🎮 Online Gaming Behavior Dashboard")

# Load your data
@st.cache_data
def load_data():
    return pd.read_csv("gaming_data.csv")  # Replace with your actual CSV file if you extract from notebook

data = load_data()

# Display raw data
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# Sample visual
st.subheader("Distribution of Game Duration")
st.bar_chart(data['game_duration'])  # Replace with actual column name from your data
