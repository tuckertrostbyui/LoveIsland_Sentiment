import streamlit as st
import glob
import pandas as pd
import plotly.express as px
from scripts.summarizer import load_summarizer, summarize_comments, classify_sentiment


st.set_page_config(
    page_title="Love Island Sentiment",  
    page_icon="🏝️", 
    layout="wide"
)


# Set Background Image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://huggingface.co/datasets/tuckertrostbyui/love_island_images/resolve/main/Beach_background.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
# Import Josefin Sans font
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Josefin+Sans:wght@400;700&display=swap');
    </style>
    """,
    unsafe_allow_html=True
)

# Custom Title and Subtitle (with only "love" bold)
st.markdown(
    """
    <h1 style='
        font-family: "Josefin Sans", sans-serif;
        font-size: 64px;
        text-align: center;
        color: white;
        text-transform: lowercase;
        font-weight: 400;
        margin-bottom: 0;
    '>
        🏝️<span style="font-weight:700;">love</span> island usa – islander breakdown🏝️
    </h1>
    <p style='
        font-family: "Josefin Sans", sans-serif;
        font-size: 24px;
        text-align: center;
        color: white;
        text-transform: lowercase;
        margin-top: 0;
    '>
        analyzing reddit sentiment of love island usa season 7
    </p>
    """,
    unsafe_allow_html=True
)



# Load the data
parquet_files = glob.glob("data/comment_updates/*.parquet")
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

islander_data = pd.read_parquet('data/islander_data/s7_islanders.parquet')

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

    # Create new layout: 1/3 snapshot | 2/3 analysis
    col_snapshot, col_analysis = st.columns([1, 2])

    # 🌴 Islander Snapshot Panel
    with col_snapshot:
        islander_row = islander_data[islander_data['name'] == selected_islander].iloc[0]

        img_filename = islander_row["filepath"].split("/")[-1]
        img_url = f"https://huggingface.co/datasets/tuckertrostbyui/love_island_images/resolve/main/images/{img_filename}"

        # Logic for optional "Exited" line
        exited_html = ""
        if str(islander_row["Status"]).lower() in ["dumped", "removed"] and pd.notna(islander_row["episode_exited"]):
            exited_html = f"<p style='font-size: 20px;'><strong>Exited:</strong> Episode {int(islander_row['episode_exited'])}</p>"

        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#5dbbf4; padding: 15px; border-radius: 10px; color: white;">
                    <img src="{img_url}" style="width:100%;" />
                    <h2 style='text-align: center; color: white; font-size: 34px;'>{islander_row['Islander']}</h2>
                    <p style='font-size: 20px;'><strong>Age:</strong> {islander_row['Age']}</p>
                    <p style='font-size: 20px;'><strong>From:</strong> {islander_row['Hometown']}</p>
                    <p style='font-size: 20px;'><strong>Status:</strong> {islander_row['Status']}</p>
                    <p style='font-size: 20px;'><strong>Entered:</strong> Episode {int(islander_row['episode_entered'])}</p>
                    {exited_html}
                </div>
                """,
                unsafe_allow_html=True
            )





            
            


    # 📈 Sentiment Analysis + KPIs
    with col_analysis:
        avg_sentiment = round(grouped['avg_sentiment'].mean(), 2)
        sentiment_label = classify_sentiment(avg_sentiment)
        comment_count = len(filtered['sentiment'])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div style="background-color:#fbe2a1; padding: 15px; border-radius: 10px; text-align:center;">
                    <p style="font-size: 14px; margin: 0;"><strong>Average Sentiment Score</strong></p>
                    <p style="font-size: 28px; margin: 0;"><strong>{avg_sentiment}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="background-color:#fbe2a1; padding: 15px; border-radius: 10px; text-align:center;">
                    <p style="font-size: 14px; margin: 0;"><strong>Average Sentiment</strong></p>
                    <p style="font-size: 28px; margin: 0;"><strong>{sentiment_label}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="background-color:#fbe2a1; padding: 15px; border-radius: 10px; text-align:center;">
                    <p style="font-size: 14px; margin: 0;"><strong>Total Comments</strong></p>
                    <p style="font-size: 28px; margin: 0;"><strong>{comment_count}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        # Group the data
        entry_ep = int(islander_row["episode_entered"])
        grouped = grouped[grouped["episode_num"] >= entry_ep]

        # Create the Plotly chart
        fig = px.line(
            grouped,
            x="episode_num",
            y="avg_sentiment",
            markers=True,
            title=f"Average Sentiment Over Time: {selected_islander}",
            labels={"episode_num": "Episode Number", "avg_sentiment": "Average Sentiment"},
            range_y=[-1, 1]
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Neutral",
            annotation_position="bottom right"
        )

        exit_status = islander_row["Status"].lower()
        exit_ep = islander_row["episode_exited"]

        if exit_status in ["dumped", "removed"] and pd.notna(exit_ep):
            fig.add_vline(
                x=int(exit_ep),
                line_dash="dash",
                line_color="red",
                annotation_text="Dumped",
                annotation_position="top right"
            )

        # Update layout to simulate a card look
        fig.update_layout(
            height=350,
            margin=dict(l=30, r=30, t=40, b=30),  
            font=dict(color="black"),
            xaxis=dict(tickmode="linear")
        )

        # Just render the chart
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)



        st.markdown(f"<h2 style='text-align: center;'> 🗣️ What are people saying about {selected_islander}?</h2>", unsafe_allow_html=True)
        if st.button('Click Here to Summarize Comments'):
            with st.spinner('Summarizing Reddit Comments...'):
                summarizer = load_summarizer()
                comments_df = filtered[['comment', 'score']]
                st.write(summarize_comments(comments_df, summarizer, selected_islander))

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
