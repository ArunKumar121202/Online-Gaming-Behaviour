import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gaming Engagement Predictor & Analysis", layout="wide")

# Styling
st.markdown("""
    <style>
    body {background-color: #0d47a1; color: white;}
    .stApp {background-color: #0d47a1; color: white;}
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label, .st-bb, .st-cb {color: white !important;}
    .stButton>button {background-color: white !important; color: #0d47a1 !important; font-weight: bold;}
    .welcome-text {color: #ffeb3b; font-weight: bold; font-size: 18px;}
    </style>
""", unsafe_allow_html=True)

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

    st.subheader("🔗 Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    im = ax_corr.imshow(corr, cmap="coolwarm")
    ax_corr.set_xticks(np.arange(len(corr.columns)))
    ax_corr.set_yticks(np.arange(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_corr.set_yticklabels(corr.columns)
    fig_corr.colorbar(im)
    st.pyplot(fig_corr)

    # Now creating 2 charts side by side
    st.markdown("## 🎮 Engagement Level & 🧑‍🤝‍🧑 Gender Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎮 Engagement Level Distribution")
        if 'Engagement_Level' in df.columns:
            fig1, ax1 = plt.subplots()
            engagement_counts = df['Engagement_Level'].value_counts().sort_index()
            bars = ax1.bar(engagement_counts.index.astype(str), engagement_counts.values, color='cyan')
            ax1.set_xlabel("Engagement Level")
            ax1.set_ylabel("Count")
            ax1.set_title("Engagement Level Distribution")

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.annotate('{}'.format(int(height)),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            st.pyplot(fig1)
        else:
            st.warning("Engagement_Level column not found.")

    with col2:
        st.subheader("🧑‍🤝‍🧑 Gender Distribution")
        if 'Gender' in df.columns:
            fig2, ax2 = plt.subplots()
            gender_counts = df['Gender'].value_counts()
            bars2 = ax2.bar(gender_counts.index, gender_counts.values, color=['lightblue', 'salmon'])
            ax2.set_xlabel("Gender")
            ax2.set_ylabel("Count")
            ax2.set_title("Gender Distribution")

            for bar in bars2:
                height = bar.get_height()
                ax2.annotate('{}'.format(int(height)),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            st.pyplot(fig2)
        else:
            st.warning("Gender column not found.")

    # Another row
    st.markdown("## 📌 Favorite Game Genre & 🌍 Player Location")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📌 Favorite Game Genre")
        if 'Favorite_Game_Genre' in df.columns:
            fig3, ax3 = plt.subplots()
            genre_counts = df['Favorite_Game_Genre'].value_counts()
            bars3 = ax3.barh(genre_counts.index, genre_counts.values, color='violet')
            ax3.set_xlabel("Count")
            ax3.set_title("Favorite Game Genre")

            for bar in bars3:
                width = bar.get_width()
                ax3.annotate('{}'.format(int(width)),
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=8)
            st.pyplot(fig3)
        else:
            st.warning("Favorite_Game_Genre column not found.")

    with col4:
        st.subheader("🌍 Player Location")
        if 'Player_Location' in df.columns:
            fig4, ax4 = plt.subplots()
            location_counts = df['Player_Location'].value_counts()
            ax4.pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            ax4.set_title("Player Location Distribution")
            st.pyplot(fig4)
        else:
            st.warning("Player_Location column not found.")

else:
    st.warning("Please upload a CSV file to continue.")
