import pandas as pd
import re
import os
import requests
from io import BytesIO
from PIL import Image
from rembg import remove, new_session
import numpy as np
from bs4 import BeautifulSoup
import face_recognition
from huggingface_hub import HfApi
from airdate_scrape import scrape_airdates

# def scrape_cast_images():
#     session = new_session("u2net_human_seg")

#     def extract_first_name(name):
#         if name == "Nicolas Vansteenberghe":
#             name = 'Nicolas "Nic" Vansteenberghe'
#         match = re.search(r'["“](.*?)["”]', name)
#         return match.group(1) if match else name.split()[0]

#     def scrape_love_island_cast(url="https://www.marieclaire.com/culture/tv-shows/love-island-usa-season-7-cast/"):
#         resp = requests.get(url)
#         resp.raise_for_status()
#         soup = BeautifulSoup(resp.text, "html.parser")
#         items = []
#         for card in soup.select("article figure, .card, .cast__item, figure"):
#             name = None
#             if card.caption:
#                 name = card.caption.get_text(strip=True)
#             else:
#                 h4 = card.find_previous(["h2", "h3", "h4"])
#                 name = h4.get_text(strip=True) if h4 else None
#             img = card.find("img")
#             img_url = img["src"] if img and img.get("src") else None
#             if name and img_url:
#                 first = extract_first_name(name)
#                 items.append({
#                     "full_name": name,
#                     "first_name": first,
#                     "name_lower": first.lower(),
#                     "image_url": img_url
#                 })
#         return pd.DataFrame(items)

#     def remove_background_from_url(image_url):
#         response = requests.get(image_url)
#         original = Image.open(BytesIO(response.content)).convert("RGBA")
#         return remove(original, session=session)

#     def crop_face_and_shoulders(pil_image, top_pad=0.8, bottom_pad=1.2, side_pad=1.5):
#         rgba_image = pil_image.convert("RGBA")
#         rgb_image = rgba_image.convert("RGB")
#         np_image = np.array(rgb_image)
#         face_locations = face_recognition.face_locations(np_image)

#         if not face_locations:
#             raise ValueError("No face detected.")

#         top, right, bottom, left = sorted(
#             face_locations,
#             key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
#             reverse=True
#         )[0]

#         face_height = bottom - top
#         face_width = right - left

#         x_start = max(0, int(left - face_width * side_pad))
#         x_end = int(right + face_width * side_pad)
#         y_start = max(0, int(top - face_height * top_pad))
#         y_end = int(bottom + face_height * bottom_pad)

#         cropped_image = rgba_image.crop((x_start, y_start, x_end, y_end))
#         return cropped_image.resize((600, 500), Image.LANCZOS)

#     cast_df = scrape_love_island_cast()
#     processed_images = []

#     for _, row in cast_df.iterrows():
#         try:
#             no_bg = remove_background_from_url(row["image_url"])
#             cropped = crop_face_and_shoulders(no_bg)
#             processed_images.append(cropped)
#         except Exception as e:
#             print(f"Error processing {row['full_name']}: {e}")
#             processed_images.append(None)

#     cast_df = cast_df[["full_name", "first_name", "name_lower"]].copy()
#     cast_df["image"] = processed_images
#     return cast_df

# def upload_images_to_hf(local_image_dir, repo_id, path_in_repo="images"):
#     from huggingface_hub import HfApi
#     api = HfApi(token=os.getenv("HF_TOKEN"))
#     api.upload_folder(
#         folder_path=local_image_dir,
#         repo_id=repo_id,
#         repo_type="dataset",
#         path_in_repo=path_in_repo,
#     )
#     print("✅ Uploaded images to Hugging Face dataset.")

# def scrape_islanders(season_num):
#     def map_day_to_episode(day, episodes_df):
#         for _, row in episodes_df.iterrows():
#             match = re.findall(r'\d+', row["Day(s)"])
#             if not match:
#                 continue
#             start_day = int(match[0])
#             end_day = int(match[-1])
#             if start_day <= day <= end_day:
#                 return row["episode_num"]
#         return pd.NA

#     def islander_episodes(islanders_df, episodes_df):
#         islanders_df["episode_entered"] = islanders_df["Entered"].apply(
#             lambda d: map_day_to_episode(d, episodes_df) if pd.notna(d) else pd.NA
#         )
#         islanders_df["episode_exited"] = islanders_df["Exited"].apply(
#             lambda d: map_day_to_episode(d, episodes_df) if pd.notna(d) else pd.NA
#         )
#         return islanders_df

#     # Step 1: Scrape base metadata
#     url = f'https://en.wikipedia.org/wiki/Love_Island_(American_TV_series)_season_{season_num}'
#     tables = pd.read_html(url)
#     islanders = pd.DataFrame(tables[1]).assign(
#         name=lambda x: x.Islander.apply(
#             lambda x: re.search(r'["“](.*?)["”]', x).group(1) if any(char in x for char in ['"', '“']) else x.split()[0]
#         ),
#         name_lower=lambda x: x.name.str.lower(),
#         Entered=lambda x: x.Entered.str.extract(r'(\d+)')[0].astype('Int64'),
#         Exited=lambda x: x.Exited.str.extract(r'(\d+)')[0].astype('Int64')
#     )

#     # Step 2: Scrape and save images locally
#     image_folder = f"data/islander_data/images"
#     os.makedirs(image_folder, exist_ok=True)

#     cast_images = scrape_cast_images()
#     cast_images["filepath"] = None

#     for i, row in cast_images.iterrows():
#         image = row["image"]
#         if image is not None:
#             filename = f"{row['name_lower'].replace(' ', '_')}_s{season_num}.png"
#             path = os.path.join(image_folder, filename)
#             image.save(path)
#             cast_images.at[i, "filepath"] = path

#     # Step 3: Upload images to HF Dataset
#     hf_repo_id = "tuckertrostbyui/love_island_images"
#     upload_images_to_hf(image_folder, hf_repo_id)

#     # Step 4: Update filepaths with hosted URLs
#     cast_images["filepath"] = cast_images["name_lower"].apply(
#         lambda name: f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main/images/{name}_s{season_num}.png"
#     )

#     # Step 5: Merge + attach episode info
#     islanders = islanders.merge(
#         cast_images[["name_lower", "filepath"]], on="name_lower", how="left"
#     )

#     episodes = scrape_airdates(season_num)
#     islanders = islander_episodes(islanders, episodes)

#     # Step 6: Save
#     output_path = f"data/islander_data/s{season_num}_islanders.parquet"
#     islanders.to_parquet(output_path, index=False)
#     print(f"✅ Saved islanders dataset: {output_path}")

#     return

# # Call it like usual
# if __name__ == "__main__":
#     scrape_islanders(7)



import pandas as pd
import re
import os
from airdate_scrape import scrape_airdates

def update_islander_parquet(season_num):
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

    # Step 1: Scrape islander table
    url = f'https://en.wikipedia.org/wiki/Love_Island_(American_TV_series)_season_{season_num}'
    tables = pd.read_html(url)
    islanders = pd.DataFrame(tables[1]).assign(
        name=lambda x: x.Islander.apply(
            lambda x: re.search(r'["“](.*?)["”]', x).group(1) if any(char in x for char in ['"', '“']) else x.split()[0]
        ),
        name_lower=lambda x: x.name.str.lower(),
        Entered=lambda x: x.Entered.str.extract(r'(\d+)')[0].astype("Int64"),
        Exited=lambda x: x.Exited.str.extract(r'(\d+)')[0].astype("Int64")
    )

    # Step 2: Load old parquet (to keep image paths)
    output_path = f"data/islander_data/s{season_num}_islanders.parquet"
    if os.path.exists(output_path):
        old = pd.read_parquet(output_path)
        if "filepath" in old.columns:
            islanders = islanders.merge(old[["name_lower", "filepath"]], on="name_lower", how="left")

    # Step 3: Scrape airdates and map episodes
    episodes = scrape_airdates(season_num)
    islanders = islander_episodes(islanders, episodes)

    # Step 4: Save updated parquet
    islanders.to_parquet(output_path, index=False)
    print(f"✅ Parquet updated at {output_path}")



update_islander_parquet(7)