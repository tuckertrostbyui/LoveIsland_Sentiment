import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("Love Island USA Season 7 - Islander Sentiment Tracker")

# Load the data
df = pd.read_csv("li_full.csv")

# Convert AirDate to datetime
df["AirDate"] = pd.to_datetime(df["AirDate"])

# Sidebar: Select islander
islanders = sorted(df["islander"].unique())
selected_islander = st.selectbox("Choose an Islander", islanders)

# Filter for selected islander and group by episode
filtered = df[df["islander"] == selected_islander]
grouped = (
    filtered.groupby(["episode_num", "AirDate"])
    .agg(avg_sentiment=("sentiment", "mean"))
    .reset_index()
    .sort_values("AirDate")
)


# Plotly line chart
fig = px.line(
    grouped,
    x="episode_num",
    y="avg_sentiment",
    markers=True,
    title=f"Average Sentiment Over Time: {selected_islander}",
    labels={"episode_num": "Episode Number", "avg_sentiment": "Average Sentiment"},
    range_y=[-1, 1]
)

fig.update_layout(xaxis=dict(tickmode="linear"))

# Show chart
st.plotly_chart(fig, use_container_width=True)
