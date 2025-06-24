import pandas as pd
import ast
from scripts.reddit_api import reddit , update_with_new_episodes
from scripts.initial_sentiment import targeted_sentiment, extract_episode_number
from scripts.airdate_scrape import scrape_airdates
from scripts.islander_scrape import scrape_islanders


def fixed_apply_sentiment(df,file_name):

    data = pd.read_parquet(df)
    comment_update = data[['comment','score','created_utc','author','episode_post_id','episode_title']]
    episode_airdates = scrape_airdates(7)
    islanders = scrape_islanders(7)['name'].to_list()

    li_update = (
        comment_update
        .assign(
            islander_sentiment=lambda x: x.comment.apply(lambda c: targeted_sentiment(c, islanders))
        )
        .assign(
            islander_sentiment=lambda df: df['islander_sentiment'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        )
        .loc[lambda df: df['islander_sentiment'].apply(lambda x: isinstance(x, dict) and len(x) > 0)]
        .assign(
            islander_sentiment_items=lambda df: df['islander_sentiment'].apply(lambda d: list(d.items()))
        )
        .explode('islander_sentiment_items')
        .assign(
            islander=lambda df: df['islander_sentiment_items'].apply(lambda x: x[0]),
            sentiment=lambda df: df['islander_sentiment_items'].apply(lambda x: x[1])
        )
        .drop(columns=['islander_sentiment_items'])
        .assign(
            episode_num=lambda x: x.episode_title.apply(extract_episode_number)
        )
        .merge(episode_airdates[['episode_num', 'airdate']], on='episode_num', how='left')
    )

    max_episode = li_update['episode_num'].max()

    li_update.to_parquet(f'{file_name}.parquet', index=False)

    return 

