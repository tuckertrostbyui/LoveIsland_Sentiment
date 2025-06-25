import os
import glob
import time
import praw
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()

# Load .env locally, skip in GitHub Actions
if os.getenv("GITHUB_ACTIONS") != "true":
    load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# --- Optimized Full Scraper ---
def scrape_all_episodes(reddit, output_folder="season7_comments", save_master=True, max_retries=3):
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Search for all Season 7 discussion threads
    season_7_posts = []
    for post in reddit.subreddit("LoveIslandUSA").search("Season 7 Episode", sort="new", limit=500):
        if "Post Episode Discussion" in post.title:
            season_7_posts.append({
                'post_id': post.id,
                'title': post.title,
                'created_utc': post.created_utc,
                'score': post.score,
                'num_comments': post.num_comments
            })

    season_df = pd.DataFrame(season_7_posts).sort_values("created_utc")
    season_df.to_parquet("season_7_episode_posts.parquet", index=False)
    print(f"✅ Found {len(season_df)} Season 7 discussion threads.")

    # Step 2: Download comments per episode with retry logic
    for idx, row in season_df.iterrows():
        post_id = row['post_id']
        title_safe = row['title'].replace('/', '_').replace(':', '').replace('"', '')
        filename = f"{output_folder}/{idx:02d}_{post_id}_comments.parquet"

        if os.path.exists(filename):
            print(f"⏩ Skipping already downloaded episode: {title_safe}")
            continue

        for attempt in range(1, max_retries + 1):
            try:
                print(f"\n🔄 Processing: {title_safe} (Attempt {attempt})")
                submission = reddit.submission(id=post_id)
                submission.comments.replace_more(limit=None, threshold=5)

                comment_data = [{
                    'comment': c.body,
                    'score': c.score,
                    'created_utc': c.created_utc,
                    'author': str(c.author),
                    'episode_post_id': post_id,
                    'episode_title': row['title']
                } for c in submission.comments.list()]

                pd.DataFrame(comment_data).to_parquet(filename, index=False)
                print(f"✅ Saved {len(comment_data)} comments for: {title_safe}")
                time.sleep(5)
                break  # success, exit retry loop

            except Exception as e:
                print(f"⚠️ Error on attempt {attempt} for {title_safe}: {e}")
                if "429" in str(e):
                    print("🛑 Rate limited. Sleeping for 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    time.sleep(10)

                if attempt == max_retries:
                    print(f"❌ Failed all {max_retries} retries for {title_safe}. Skipping to next.")

    # Step 3: Combine into master file
    if save_master:
        all_files = glob.glob(f"{output_folder}/*_comments.parquet")
        dfs = [pd.read_csv(file) for file in all_files]
        master_df = pd.concat(dfs, ignore_index=True)
        master_df.to_parquet("season7_all_episode_comments.parquet", index=False)
        print(f"\n📦 Master file created with {len(master_df)} total comments.")

# --- Incremental Update Function ---
def update_with_new_episodes(reddit, output_folder="data/season7_comments", max_retries=3):
    os.makedirs(output_folder, exist_ok=True)

    # Get a set of already-downloaded post IDs from filenames
    existing_files = {
        re.search(r'_(\w+)_comments\.parquet$', f).group(1)
        for f in glob.glob(f"{output_folder}/*_comments.parquet")
        if re.search(r'_(\w+)_comments\.parquet$', f)
    }

    # Step 1: Search for new Season 7 posts
    new_posts = []
    for post in reddit.subreddit("LoveIslandUSA").search("Season 7 Episode", sort="new", limit=500):
        if "Post Episode Discussion" in post.title and post.id not in existing_files:
            new_posts.append({
                'post_id': post.id,
                'title': post.title,
                'created_utc': post.created_utc,
                'score': post.score,
                'num_comments': post.num_comments
            })

    if not new_posts:
        print("✅ No new episodes to update.")
        return pd.DataFrame()  # Return empty DataFrame

    new_df = pd.DataFrame(new_posts).sort_values("created_utc")
    all_new_comments = []

    print(f"🆕 Found {len(new_df)} new episodes. Downloading now...")

    for idx, row in new_df.iterrows():
        post_id = row['post_id']
        title_safe = row['title'].replace('/', '_').replace(':', '').replace('"', '')
        filename = f"{output_folder}/{len(existing_files)+idx:02d}_{post_id}_comments.parquet"

        for attempt in range(1, max_retries + 1):
            try:
                print(f"\n🔄 Processing: {title_safe} (Attempt {attempt})")
                submission = reddit.submission(id=post_id)
                submission.comments.replace_more(limit=None, threshold=5)

                comment_data = [{
                    'comment': c.body,
                    'score': c.score,
                    'created_utc': c.created_utc,
                    'author': str(c.author),
                    'episode_post_id': post_id,
                    'episode_title': row['title']
                } for c in submission.comments.list()]

                df_episode = pd.DataFrame(comment_data)
                df_episode.to_parquet(filename, index=False)
                all_new_comments.append(df_episode)

                print(f"✅ Saved {len(comment_data)} comments for: {title_safe}")
                time.sleep(3)
                break

            except Exception as e:
                print(f"⚠️ Error on attempt {attempt} for {title_safe}: {e}")
                if "429" in str(e):
                    print("🛑 Rate limited. Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    time.sleep(10)

                if attempt == max_retries:
                    print(f"❌ Failed all {max_retries} retries for {title_safe}. Skipping.")

    # Combine and return all new data
    if all_new_comments:
        master_df = pd.concat(all_new_comments, ignore_index=True)
        print(f"\n📦 Returning DataFrame with {len(master_df)} new comments.")
        return master_df
    else:
        print("📭 No new comments were successfully downloaded.")
        return pd.DataFrame()
