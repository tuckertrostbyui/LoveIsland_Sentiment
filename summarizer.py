import streamlit as st
from transformers import pipeline
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API Key
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')


@st.cache_resource
def load_summarizer():
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def prepare_text_for_summary(comments, max_chars=3000):
    cleaned = [
        c.strip().replace('\n', ' ') for c in comments
        if len(c.strip()) > 30  # remove super short ones
    ]
    combined = " ".join(cleaned[:50])
    return combined[:max_chars]


def summarize_comments(comments,summarizer):
    if not comments:
        return "No comments available."

    # Clean and join text
    text = prepare_text_for_summary(comments)

    # Add framing prompt
    prompt = "Summarize what people are saying about this person:\n " + text

    try:
        response = summarizer.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during summarization: {e}"
