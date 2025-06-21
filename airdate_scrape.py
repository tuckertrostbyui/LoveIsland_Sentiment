import pandas as pd
import datetime as dt

def scrape_airdates(season_num):
    url = f'https://en.wikipedia.org/wiki/Love_Island_(American_TV_series)_season_{season_num}'
    tables = pd.read_html(url)
    episodes = pd.DataFrame(tables[3])

    episodes_clean = episodes[['Title','Original release date [19]']]\
        .loc[~episodes['Title'].str.contains("Week", case=False, na=False)]\
        .rename(columns={'Title':'episode','Original release date [19]':'airdate'})\
        .assign(
            episode_num = lambda x: x.episode.str.extract(r'(\d+)').astype(int),
            airdate = lambda x: pd.to_datetime(x.airdate)
        )
    return episodes_clean