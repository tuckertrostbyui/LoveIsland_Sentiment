import pandas as pd
import re
from airdate_scrape import scrape_airdates
import os

def scrape_cast_images():
    import pandas as pd
    from io import BytesIO
    import requests
    from PIL import Image
    from rembg import remove, new_session
    import numpy as np
    from bs4 import BeautifulSoup
    import face_recognition

    # Load a more robust model for human background removal
    session = new_session("u2net_human_seg")

    def extract_first_name(name):
        if name == "Nicolas Vansteenberghe":
            name = 'Nicolas "Nic" Vansteenberghe'
        match = re.search(r'["“](.*?)["”]', name)
        return match.group(1) if match else name.split()[0]

    def scrape_love_island_cast(url="https://www.marieclaire.com/culture/tv-shows/love-island-usa-season-7-cast/"):
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = []
        for card in soup.select("article figure, .card, .cast__item, figure"):
            name = None
            if card.caption:
                name = card.caption.get_text(strip=True)
            else:
                h4 = card.find_previous(["h2", "h3", "h4"])
                name = h4.get_text(strip=True) if h4 else None
            img = card.find("img")
            img_url = img["src"] if img and img.get("src") else None
            if name and img_url:
                first = extract_first_name(name)
                items.append({
                    "full_name": name,
                    "first_name": first,
                    "name_lower": first.lower(),
                    "image_url": img_url
                })
        return pd.DataFrame(items)

    def remove_background_from_url(image_url):
        response = requests.get(image_url)
        original = Image.open(BytesIO(response.content)).convert("RGBA")
        return remove(original, session=session)

    def crop_face_and_shoulders(pil_image, top_pad=0.8, bottom_pad=1.2, side_pad=1.5):
        rgba_image = pil_image.convert("RGBA")
        rgb_image = rgba_image.convert("RGB")
        np_image = np.array(rgb_image)
        face_locations = face_recognition.face_locations(np_image)

        if not face_locations:
            raise ValueError("No face detected.")

        top, right, bottom, left = sorted(
            face_locations,
            key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
            reverse=True
        )[0]

        face_height = bottom - top
        face_width = right - left

        x_start = max(0, int(left - face_width * side_pad))
        x_end = int(right + face_width * side_pad)
        y_start = max(0, int(top - face_height * top_pad))
        y_end = int(bottom + face_height * bottom_pad)

        cropped_image = rgba_image.crop((x_start, y_start, x_end, y_end))
        return cropped_image.resize((600, 500), Image.LANCZOS)

    # ---- Main Workflow ---- #
    cast_df = scrape_love_island_cast()
    processed_images = []

    for _, row in cast_df.iterrows():
        try:
            no_bg = remove_background_from_url(row["image_url"])
            cropped = crop_face_and_shoulders(no_bg)
            processed_images.append(cropped)
        except Exception as e:
            print(f"Error processing {row['full_name']}: {e}")
            processed_images.append(None)

    cast_df = cast_df[["full_name", "first_name", "name_lower"]].copy()
    cast_df["image"] = processed_images
    return cast_df


def scrape_islanders(season_num):
    def map_day_to_episode(day, episodes_df):
        for _, row in episodes_df.iterrows():
            match = re.findall(r'\d+', row["Day(s)"])
            if not match:
                continue
            start_day = int(match[0])
            end_day = int(match[-1])
            if start_day <= day <= end_day:
                return row["episode_num"]
        return pd.NA

    def islander_episodes(islanders_df, episodes_df):
        islanders_df["episode_entered"] = islanders_df["Entered"].apply(
            lambda d: map_day_to_episode(d, episodes_df) if pd.notna(d) else pd.NA
        )
        islanders_df["episode_exited"] = islanders_df["Exited"].apply(
            lambda d: map_day_to_episode(d, episodes_df) if pd.notna(d) else pd.NA
        )
        return islanders_df

    # Scrape Wikipedia table
    url = f'https://en.wikipedia.org/wiki/Love_Island_(American_TV_series)_season_{season_num}'
    tables = pd.read_html(url)
    islanders = pd.DataFrame(tables[1]).assign(
        name=lambda x: x.Islander.apply(
            lambda x: re.search(r'["“](.*?)["”]', x).group(1) if any(char in x for char in ['"', '“']) else x.split()[0]
        ),
        name_lower=lambda x: x.name.str.lower(),
        Entered=lambda x: x.Entered.str.extract(r'(\d+)')[0].astype('Int64'),
        Exited=lambda x: x.Exited.str.extract(r'(\d+)')[0].astype('Int64')
    )

    # Scrape and store images
    image_folder = f"data/islander_data/images"
    os.makedirs(image_folder, exist_ok=True)

    cast_images = scrape_cast_images()
    cast_images["filepath"] = None

    for i, row in cast_images.iterrows():
        image = row["image"]
        if image is not None:
            filename = f"{row['name_lower'].replace(' ', '_')}_s{season_num}.png"
            path = os.path.join(image_folder, filename)
            image.save(path)
            cast_images.at[i, "filepath"] = path

    # Merge image file paths into islander data
    islanders = islanders.merge(
        cast_images[["name_lower", "filepath"]], on="name_lower", how="left"
    )

    # Attach episode entry/exit numbers
    episodes = scrape_airdates(season_num)
    islanders = islander_episodes(islanders, episodes)

    # Save to parquet
    output_path = f"data/islander_data/s{season_num}_islanders.parquet"
    islanders.to_parquet(output_path, index=False)

    return 

scrape_islanders(7)
