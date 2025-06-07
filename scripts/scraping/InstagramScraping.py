import os
import time
import random
import json
from typing import List, Dict, Any
import pandas as pd
from instagrapi import Client
from dotenv import load_dotenv

load_dotenv()

def _random_delay(min_delay: float = 0.3, max_delay: float = 1.3) -> None:
    """Sleep a random amount to mimic human-like pauses."""
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)


def _serialize_media(media, cl: Client, n_comments: int) -> Dict[str, Any]:
    """Turn an instagrapi media object into a flat python dict."""
    data = {
        "media_pk": media.pk,
        "code": media.code,
        "media_type": media.media_type,
        "likes": media.like_count,
        "views": getattr(media, "view_count", None),
        "comment_count": media.comment_count,
        "taken_at": media.taken_at.isoformat() if media.taken_at else None,
        "caption_text": media.caption_text,
        "owner_username": media.user.username,
        "owner_pk": media.user.pk,
        "thumbnail_url": media.thumbnail_url,
    }

    # fetch comments (quietly)
    if n_comments > 0 and media.comment_count:
        try:
            comments = cl.media_comments(media.id, amount=n_comments)
            data["comments"] = [c.text for c in comments]
        except Exception as e:
            data["comments"] = []
            data["comment_error"] = str(e)

    return data


def scrape_ig_car_posts(
    car_name: str,
    username: str,
    password: str,
    n_top: int = 100,
    n_recent: int = 0,
    n_comments: int = 15,
    save: bool = True,
    save_path: str = "./",
) -> pd.DataFrame:
    """
    Scrape Instagram hashtag posts for a classic/exotic car.

    Parameters
    ----------
    car_name : str
        Car model to scrape (e.g. "Ferrari F40").  Used as the hashtag and as
        the primary key column (tag).
    username, password : str
        Instagram credentials (ideally load from environment variables).
    n_top : int, default 100
        Number of *Top* posts to fetch.
    n_recent : int, default 0
        Number of *Recent* posts to fetch (set >0 to include).
    n_comments : int, default 15
        Maximum number of comments per post to capture.
    save : bool, default True
        If True, write a JSON and CSV to `save_path`.
    save_path : str, default "./"
        Directory path (file names are auto-generated).

    Returns
    -------
    pandas.DataFrame
        One row per media post with flattened metadata & comments list.
    """
    tag = car_name.replace(" ", "").lower()

    # --- 1. Login ------------------
    cl = Client()
    cl.login(username, password)

    # --- 2. Scrape posts ------------
    posts: List[Dict[str, Any]] = []

    if n_top:
        top_medias = cl.hashtag_medias_top(tag, amount=n_top)
        for m in top_medias:
            posts.append(_serialize_media(m, cl, n_comments))
            _random_delay()

    if n_recent:
        recent_medias = cl.hashtag_medias_recent(tag, amount=n_recent)
        for m in recent_medias:
            posts.append(_serialize_media(m, cl, n_comments))
            _random_delay()

    # --- 3. Build DataFrame ---------
    df = pd.DataFrame(posts)
    df.insert(0, "tag", car_name)  # primary key / partition field

    # --- 4. Save (optional) ---------
    if save:
        os.makedirs(save_path, exist_ok=True)
        stem = f"hashtag_{tag}_posts"
        df.to_json(os.path.join(save_path, f"{stem}.json"), orient="records", force_ascii=False, indent=2)
        df.to_csv(os.path.join(save_path, f"{stem}.csv"), index=False)

    return df