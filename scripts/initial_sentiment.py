import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import ast
import datetime as dt
from airdate_scrape import scrape_airdates
from islander_scrape import scrape_islanders
import spacy

# Load Model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Sentiment Function
def get_sentiment_score(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=1)
    return {
        'negative': scores[0][0].item(),
        'neutral': scores[0][1].item(),
        'positive': scores[0][2].item(),
        'compound': scores[0][2].item() - scores[0][0].item()
    }




nlp = spacy.load("en_core_web_sm")

def targeted_sentiment(comment, islanders):
    islander_sentiment = {}
    doc = nlp(comment)

    for sent in doc.sents:
        sentence_text = sent.text
        # Split each sentence into smaller chunks by contrastive conjunctions and commas
        raw_chunks = re.split(r'\bbut\b|\band\b|,', sentence_text, flags=re.IGNORECASE)

        for chunk in raw_chunks:
            if len(chunk.strip()) == 0:
                continue
            chunk_doc = nlp(chunk)
            result = get_sentiment_score(chunk)
            chunk_lower = chunk.lower()

            for name in islanders:
                if name.lower() in chunk_lower:
                    if name not in islander_sentiment:
                        islander_sentiment[name] = []
                    islander_sentiment[name].append(result["compound"])

    return {name: np.mean(scores) for name, scores in islander_sentiment.items()}


# Load Data

# all_comments = pd.read_parquet('../data/season7_all_episode_comments.parquet')
# episode_airdates = scrape_airdates(7)

islanders = scrape_islanders(7)['name'].to_list()

def extract_episode_number(title):
    match = re.search(r'Episode (\d+)', title)
    if match:
        return int(match.group(1))
    return None

# li_initial = all_comments\
#     .assign(
#         islander_sentiment = lambda x: x.comment.apply(lambda x: targeted_sentiment(x,islanders))
#     )\
#     .assign(islander_sentiment=lambda df: df['islander_sentiment'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x))\
#     .loc[lambda df: df['islander_sentiment'].apply(lambda x: isinstance(x, dict) and len(x) > 0)]\
#     .assign(islander_sentiment_items=lambda df: df['islander_sentiment'].apply(lambda d: list(d.items())))\
#     .explode('islander_sentiment_items')\
#     .assign(**{
#         'islander': lambda df: df['islander_sentiment_items'].apply(lambda x: x[0]),
#         'sentiment': lambda df: df['islander_sentiment_items'].apply(lambda x: x[1])
#     })\
#     .drop(columns=['islander_sentiment_items'])\
#     .assign(
#         episode_num = lambda x: x.episode_title.apply(extract_episode_number)
#     )\
#     .merge(episode_airdates[['episode_num','airdate']],on='episode_num',how='left')

# li_initial.to_parquet('li_initial.parquet',index=False)