import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.summarizer import load_summarizer, summarize_comments, classify_sentiment

st.set_page_config(
    page_title="Love Island Sentiment",  
    page_icon="🏝️", 
    layout="wide"
)

# Title
st.markdown("<h1 style='text-align: center; color: Black;'>🏝️Love Island USA - Islander Breakdown🏝️</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: gray;'>Analysing Reddit Sentiment of Love Island USA Season 7</h6>", unsafe_allow_html=True)

# Load the data
df = pd.read_parquet("data/li_initial.parquet")

# Convert AirDate to datetime
df["airdate"] = pd.to_datetime(df["airdate"])

tab1, tab2 = st.tabs(['Dashboard','Info'])

with tab1:
    # Sidebar: Select islander
    islanders = sorted(df["islander"].unique())
    selected_islander = st.selectbox("Choose an Islander", islanders)

    # Filter for selected islander and group by episode
    filtered = df[df["islander"] == selected_islander]
    grouped = (
        filtered.groupby(["episode_num", "airdate"])
        .agg(avg_sentiment=("sentiment", "mean"))
        .reset_index()
        .sort_values("airdate")
        .assign(
            sentiment_classification = lambda x: x.avg_sentiment.apply(classify_sentiment)
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_sentiment = grouped['avg_sentiment'].mean()
        with st.container(border=True):
            st.metric(label='Average Sentiment Score', value=round(avg_sentiment,2))

    with col2:
        sentiment_label = classify_sentiment(avg_sentiment)
        with st.container(border=True):
            st.metric(label='Average Sentiment', value=sentiment_label)

    with col3:
        comment_count = len(filtered['sentiment'])
        with st.container(border=True):
            st.metric(label='Total Comments', value=comment_count)

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

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral", annotation_position="bottom right")


    fig.update_layout(xaxis=dict(tickmode="linear"))

    # Show chart
    st.plotly_chart(fig, use_container_width=True)

    # Summarizer
    st.markdown(f"<h2 style='text-align: center; color: Black;'>What are people saying about {selected_islander}?</h2>", unsafe_allow_html=True)

    summarizer = load_summarizer()

    comments_df = df[df['islander'] == selected_islander][['comment', 'score']]

    if st.button('Summarize Comments'):
        with st.spinner('Summarizing Reddit Comments...'):
            st.write(summarize_comments(comments_df, summarizer,selected_islander))


with tab2:
    st.markdown("""
## ℹ️ About This Dashboard

### 📊 What is a Sentiment Score?

Each Reddit comment about a Love Island USA contestant is analyzed to measure its **sentiment** — the emotional tone of the message.  
The sentiment score ranges from **-1 to 1**:

- **+1** → Very positive (e.g., admiration, excitement)
- **0** → Neutral (e.g., factual or emotionless)
- **-1** → Very negative (e.g., criticism, dislike)

For each islander, sentiment scores are averaged by episode to track how public perception changes over time.

---

### 🔍 How Was the Data Collected?

The comments were collected from **official Reddit discussion threads** for each episode of **Love Island USA Season 7**.  
Using the **Reddit API**, the app pulls comments that mention each islander by name.

All comments were cleaned and filtered to remove:
- Spam
- Irrelevant discussions
- Duplicates

This ensures that the sentiment analysis focuses only on meaningful feedback from the community.

---

### 🤖 How Are the Comment Summaries Generated?

When you click **"Summarize Comments"**, the app uses the **Gemini API** (Google’s large language model) to generate a short, natural-language summary.

This summary reflects:
- Key discussion topics
- Recurring opinions
- Overall tone of the conversation

It helps answer the *why* behind the sentiment score, giving you a quick understanding of what people are actually saying.

""")
