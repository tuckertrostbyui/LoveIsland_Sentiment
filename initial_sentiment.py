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

# Targeted Sentiment Function
def targeted_sentiment(comment, islanders):
    islander_sentiment = {}
    for sentence in sent_tokenize(comment):
        for name in islanders:
            if re.search(rf'\b{name}\b', sentence, re.IGNORECASE):
                result = get_sentiment_score(sentence)
                if name not in islander_sentiment:
                    islander_sentiment[name] = []
                islander_sentiment[name].append(result['compound'])
    return {name: np.mean(scores) for name, scores in islander_sentiment.items()}

# Load Data

all_comments = pd.read_csv('season7_all_episode_comments.csv')
episode_airdates = scrape_airdates(7)

islanders = ['Chelley','Olandria','Huda','Ace','Nic','Taylor','Jeremiah','Austin','Charlie','Cierra','Hannah','Amaya','Pepe','Jalen','Iris','Yulissa','Belle-A']

def extract_episode_number(title):
    match = re.search(r'Episode (\d+)', title)
    if match:
        return int(match.group(1))
    return None

li_initial = all_comments\
    .assign(
        islander_sentiment = lambda x: x.comment.apply(lambda x: targeted_sentiment(x,islanders))
    )\
    .assign(islander_sentiment=lambda df: df['islander_sentiment'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x))\
    .loc[lambda df: df['islander_sentiment'].apply(lambda x: isinstance(x, dict) and len(x) > 0)]\
    .assign(islander_sentiment_items=lambda df: df['islander_sentiment'].apply(lambda d: list(d.items())))\
    .explode('islander_sentiment_items')\
    .assign(**{
        'islander': lambda df: df['islander_sentiment_items'].apply(lambda x: x[0]),
        'sentiment': lambda df: df['islander_sentiment_items'].apply(lambda x: x[1])
    })\
    .drop(columns=['islander_sentiment_items'])\
    .assign(
        episode_num = lambda x: x.episode_title.apply(extract_episode_number)
    )\
    .merge(episode_airdates[['episode_num','airdate']],on='episode_num',how='left')

li_initial.to_csv('li_initial.csv',index=False)