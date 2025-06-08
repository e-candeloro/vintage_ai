import os
import time
import random
import json
from typing import List, Dict, Any
import pandas as pd
from instagrapi import Client
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel

load_dotenv()

n_top = 100
n_recent = 0
n_comments = 15
save = True

class Metrics(BaseModel):
    num_comments: int
    avg_sentiment_score: float
    most_common_sentiment: str
    likes: int
    shares: Optional[int]
    plays: Optional[int]
    collections: Optional[int]
    engagement_score: Optional[int]
    overall_sentiment_score: int

def random_delay(min_delay: float = 0.1, max_delay: float = 1) -> None:
    delay = random.uniform(min_delay, max_delay)
    print(f"Sleeping {delay:4.2f}s …")
    time.sleep(delay)

def scraping(target: str):
    # Fetch top 100 post of the target hashtag
    print(f"Fetching posts for hashtag #{target.replace(" ", "").lower()}")
    medias = cl.hashtag_medias_top(target.replace(" ", "").lower(), amount=n_top)
    posts_data = []
    for media in medias:
        print(f"Processing media ID: {media.pk}")
        random_delay()  # Delay to mimic human-like behavior
        post_info = {
            "likes": media.like_count,
            "views": media.view_count,
            "posting_date": media.taken_at.isoformat() if hasattr(media, 'taken_at') else None,
            "caption_text": media.caption_text, 
            "comments_count": media.comment_count,
            "comments": []
        }
        # Get up to 15 comments
        try:
            comments = cl.media_comments(media.id, amount=n_comments)
            post_info["comments"] = [comment.text for comment in comments]
        except Exception as e:
            print(f"Could not fetch comments for media {media.pk}: {e}")
            cl = Client()
            cl.login(os.getenv(IG_USER), os.getenv(IG_PASS))
            comments = cl.media_comments(media.id, amount=n_comments)
            post_info["comments"] = [comment.text for comment in comments]
        medias = cl.hashtag_medias_recent(target.replace(" ", "").lower(), amount=n_recent)
    for media in medias:
        print(f"Processing media ID: {media.pk}")
        random_delay()  # Delay to mimic human-like behavior
        post_info = {
            "likes": media.like_count,
            "views": media.view_count,
            "posting_date": media.taken_at.isoformat() if hasattr(media, 'taken_at') else None,
            "caption_text": media.caption_text, 
            "comments_count": media.comment_count,
            "comments": []
        }
        # Get up to 15 comments
        try:
            comments = cl.media_comments(media.id, amount=n_comments)
            post_info["comments"] = [comment.text for comment in comments]
        except Exception as e:
            print(f"Could not fetch comments for media {media.pk}: {e}")
            cl = Client()
            cl.login(os.getenv(IG_USER), os.getenv(IG_PASS))
            comments = cl.media_comments(media.id, amount=n_comments)
            post_info["comments"] = [comment.text for comment in comments]
        posts_data.append(post_info)
    # Save data to JSON
    if save:
        output_file = f"hashtag_{target}_posts.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(posts_data, f, indent=4, ensure_ascii=False)
        print(f"Saved data to {output_file}")
    return posts_data
def _clean(item: dict) -> dict:
    def _i(x):
        try:
            return int(x)
        except:
            return 0
        comms = item.get("comments")
        comms.append(item.get("caption_text"))
        return dict(
            likes=_i(item.get("likes")),
            date=_i(item.get("posting_date")),
            comments_count=_i(item.get("comments_count"))
            comments=comms,
        )


# --- 1. Login ------------------
cl = Client()
cl.login(os.getenv(IG_USER), os.getenv(IG_PASS))

# --- 2. Scrape posts ------------
target = input("Select target hash: ")
scrap_data = scraping(target)
clean_data = []
for post in scrap_data:
    clean_data.append(_clean(post))

