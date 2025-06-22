import os
import glob
import time
import praw
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

reddit_client_id = os.getenv('reddit_client_id')
reddit_client_secret = os.getenv('reddit_client_secret')
reddit_user_agent = os.getenv('reddit_user_agent')

# Initialize Reddit API with PRAW
reddit = praw.Reddit(
    client_id = reddit_client_id,
    client_secret = reddit_client_secret,
    user_agent = reddit_user_agent
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
    season_df.to_csv("season_7_episode_posts.csv", index=False)
    print(f"✅ Found {len(season_df)} Season 7 discussion threads.")

    # Step 2: Download comments per episode with retry logic
    for idx, row in season_df.iterrows():
        post_id = row['post_id']
        title_safe = row['title'].replace('/', '_').replace(':', '').replace('"', '')
        filename = f"{output_folder}/{idx:02d}_{post_id}_comments.csv"

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

                pd.DataFrame(comment_data).to_csv(filename, index=False)
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
        all_files = glob.glob(f"{output_folder}/*_comments.csv")
        dfs = [pd.read_csv(file) for file in all_files]
        master_df = pd.concat(dfs, ignore_index=True)
        master_df.to_csv("season7_all_episode_comments.csv", index=False)
        print(f"\n📦 Master file created with {len(master_df)} total comments.")

# --- Incremental Update Function ---
def update_with_new_episodes(output_folder="season7_comments"):
    existing_files = {f.split('/')[-1].split('_')[1] for f in glob.glob(f"{output_folder}/*_comments.csv")}

    season_7_posts = []
    for post in reddit.subreddit("LoveIslandUSA").search("Season 7 Episode", sort="new", limit=500):
        if "Post Episode Discussion" in post.title:
            if post.id not in existing_files:
                season_7_posts.append({
                    'post_id': post.id,
                    'title': post.title,
                    'created_utc': post.created_utc,
                    'score': post.score,
                    'num_comments': post.num_comments
                })

    if not season_7_posts:
        print("✅ No new episodes to update.")
        return

    new_df = pd.DataFrame(season_7_posts).sort_values("created_utc")
    season_df = pd.read_csv("season_7_episode_posts.csv")
    updated_df = pd.concat([season_df, new_df], ignore_index=True).drop_duplicates(subset="post_id")
    updated_df.to_csv("season_7_episode_posts.csv", index=False)

    print(f"🆕 Found {len(new_df)} new episodes. Downloading now...")
    for idx, row in new_df.iterrows():
        post_id = row['post_id']
        title_safe = row['title'].replace('/', '_').replace(':', '').replace('"', '')
        filename = f"{output_folder}/{len(season_df)+idx:02d}_{post_id}_comments.csv"

        try:
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

            pd.DataFrame(comment_data).to_csv(filename, index=False)
            print(f"✅ Saved {len(comment_data)} comments for: {title_safe}")
            time.sleep(3)

        except Exception as e:
            print(f"❌ Error downloading new episode {post_id}: {e}")

    # Update master file
    all_files = glob.glob(f"{output_folder}/*_comments.csv")
    dfs = [pd.read_csv(file) for file in all_files]
    master_df = pd.concat(dfs, ignore_index=True)
    master_df.to_csv("season7_all_episode_comments.csv", index=False)
    print(f"\n📦 Master file updated with {len(master_df)} total comments.")


scrape_all_episodes(reddit)