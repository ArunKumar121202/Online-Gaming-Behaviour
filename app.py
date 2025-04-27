import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Online Gaming Behavior Insights",
    page_icon="🎮",
    layout="wide",
)

# --- CUSTOM CSS for styling ---
st.markdown("""
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    /* Main background */
    .stApp {
        background: linear-gradient(to right, #1f1c2c, #928DAB);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: white;
        border-radius: 8px;
        padding: 8px;
        color: black;
    }
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("online_gaming_behavior_dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

df = load_data()

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/747/747376.png", width=100)
st.sidebar.title(f"👋 Welcome, Arun!")
st.sidebar.markdown("### Choose Section")

section = st.sidebar.selectbox(
    "Go to",
    ("Overview", "Player Demographics", "Player Activity", "Spending Patterns")
)

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>🎯 Online Gaming Behavior - Insights Dashboard 🎮</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- OVERVIEW SECTION ---
if section == "Overview":
    total_players = df.shape[0]
    avg_age = round(df['Age'].mean(), 2) if 'Age' in df.columns else 0
    males = df[df['Gender'] == 'Male'].shape[0] if 'Gender' in df.columns else 0
    females = df[df['Gender'] == 'Female'].shape[0] if 'Gender' in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'>👥<br><h3>Total Players</h3><h2>{}</h2></div>".format(total_players), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>🎂<br><h3>Average Age</h3><h2>{}</h2></div>".format(avg_age), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>♂️<br><h3>Males</h3><h2>{}</h2></div>".format(males), unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'>♀️<br><h3>Females</h3><h2>{}</h2></div>".format(females), unsafe_allow_html=True)

    st.markdown("## ")
    st.markdown("### 📊 Summary Plots")

    fig_age = px.histogram(df, x="Age", nbins=30, title="Age Distribution", color_discrete_sequence=["#00BFFF"])
    fig_gender = px.bar(df['Gender'].value_counts().reset_index(), x='index', y='Gender',
                        title="Gender Distribution", color='index', color_discrete_sequence=px.colors.qualitative.Pastel)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(fig_age, use_container_width=True)
    with col6:
        st.plotly_chart(fig_gender, use_container_width=True)

# --- PLAYER DEMOGRAPHICS ---
elif section == "Player Demographics":
    st.markdown("## 🎮 Player Demographics")

    fig_location = px.bar(df['Location'].value_counts().head(10).reset_index(),
                         x='index', y='Location',
                         title="Top 10 Locations", color='index', color_discrete_sequence=px.colors.qualitative.Bold)

    fig_age_gender = px.histogram(df, x="Age", color="Gender", barmode="overlay",
                                  title="Age Distribution by Gender",
                                  color_discrete_sequence=["#6a0dad", "#ff69b4"])

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(fig_location, use_container_width=True)
    with col8:
        st.plotly_chart(fig_age_gender, use_container_width=True)

# --- PLAYER ACTIVITY ---
elif section == "Player Activity":
    st.markdown("## 🕹 Player Activity")

    fig_sessions = px.histogram(df, x="SessionsPerWeek", nbins=30, title="Sessions Per Week Distribution",
                                color_discrete_sequence=["#ff7f0e"])

    fig_hours = px.histogram(df, x="Hours_Played_Per_Week", nbins=30, title="Hours Played Per Week Distribution",
                             color_discrete_sequence=["#2ca02c"])

    col9, col10 = st.columns(2)
    with col9:
        st.plotly_chart(fig_sessions, use_container_width=True)
    with col10:
        st.plotly_chart(fig_hours, use_container_width=True)

# --- SPENDING PATTERNS ---
elif section == "Spending Patterns":
    st.markdown("## 💰 Spending Patterns")

    if 'Spending_Score' in df.columns:
        fig_spending = px.histogram(df, x="Spending_Score", nbins=30, title="Spending Score Distribution",
                                    color_discrete_sequence=["#d62728"])
        st.plotly_chart(fig_spending, use_container_width=True)
    else:
        st.warning("Spending Score data not available.")
