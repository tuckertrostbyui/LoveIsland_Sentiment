import pandas as pd
import re

def scrape_islanders(season_num):
    url = f'https://en.wikipedia.org/wiki/Love_Island_(American_TV_series)_season_{season_num}'
    tables = pd.read_html(url)
    islanders = pd.DataFrame(tables[1])

    islanders['name'] = islanders['Islander'].apply(
        lambda x: re.search(r'["“](.*?)["”]', x).group(1) if any(char in x for char in ['"', '“']) else x.split()[0]
    )

    return islanders