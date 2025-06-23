import pandas as pd
import datetime as dt

def scrape_airdates(season_num):
    url = f'https://en.wikipedia.org/wiki/Love_Island_(American_TV_series)_season_{season_num}'
    tables = pd.read_html(url)
    episodes = pd.DataFrame(tables[3])

    # Clean column names
    episodes.columns = episodes.columns.str.strip()

    # Dynamically find the release date column
    release_col = [col for col in episodes.columns if 'original release date' in col.lower()]
    if not release_col:
        raise ValueError("❌ Could not find a column containing 'Original release date'")
    
    episodes_clean = (
        episodes[['Title', release_col[0]]]
        .loc[~episodes['Title'].str.contains("Week", case=False, na=False)]
        .rename(columns={'Title': 'episode', release_col[0]: 'airdate'})
        .assign(
            episode_num=lambda x: x.episode.str.extract(r'(\d+)').astype(int),
            airdate=lambda x: pd.to_datetime(x.airdate)
        )
    )

    return episodes_clean
