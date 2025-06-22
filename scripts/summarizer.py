import streamlit as st
from transformers import pipeline
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

# Load API Key
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')


@st.cache_resource
def load_summarizer():
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def prepare_text_for_summary(df, max_comments=50, max_chars=3000):
    # Ensure necessary columns are present
    if not {'comment', 'score'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'comment' and 'score' columns.")

    # Remove short comments
    filtered = df[df['comment'].str.len() > 30]

    # Sort by upvotes (score) in descending order and limit
    top_comments = (
        filtered
        .sort_values(by='score', ascending=False)
        .head(max_comments)
    )

    # Clean and combine
    cleaned = top_comments['comment'].str.strip().str.replace('\n', ' ', regex=False)
    combined = " ".join(cleaned)
    return combined[:max_chars]


def summarize_comments(comments, summarizer,selected_islander):
    # Check for empty DataFrame
    if comments.empty:
        return "No comments available."

    # Prepare summary text from top comments
    text = prepare_text_for_summary(comments)

    # Frame the prompt
    prompt = f"Summarize what people are saying about {selected_islander}:\n{text}"

    # Generate summary using Gemini
    try:
        response = summarizer.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during summarization: {e}"


def classify_sentiment(score):
  if score >= 0.05:
    return 'Positive'
  elif score <= -0.05:
    return 'Negative'
  else:
    return 'Neutral'