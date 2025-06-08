import os
import time
import random
import pandas as pd
from instagrapi import Client

# ───────────────────────────────
# 1.  LOGIN  (use env vars or replace)
# ───────────────────────────────
cl = Client()
cl.login()

# ───────────────────────────────
# 2.  CONFIG
# ───────────────────────────────
HASHTAG = "FerrariTestarossa"
TOP_MEDIA_AMOUNT = 10  # posts to analyse
MAX_COMMENTS = 100  # per post
TOP_COMMENTS = 10  # keep the most-liked N comments

# ───────────────────────────────
# 3.  HELPERS
# ───────────────────────────────


def random_delay(min_delay: float = 1, max_delay: float = 5) -> None:
    delay = random.uniform(min_delay, max_delay)
    print(f"Sleeping {delay:4.2f}s …")
    time.sleep(delay)


def parse_insights(raw: dict) -> dict:
    """
    Flattens the structure returned by cl.insights_media(...)
    so we can safely call metrics['like_count'] etc.
    Handles both list-based and dict-based formats.
    """
    metrics = {}
    for k, v in raw.items():
        if isinstance(v, list):
            metrics[k] = v[0].get("value", 0) if v else 0
        elif isinstance(v, dict) and "values" in v:
            metrics[k] = v["values"][0].get("value", 0)
        else:
            metrics[k] = v or 0
    return metrics


def safe_caption(media) -> str:
    return getattr(media, "caption_text", "") or (
        media.caption and media.caption.text or ""
    )


# ───────────────────────────────
# 4.  MAIN
# ───────────────────────────────
post_data = []

for media in cl.hashtag_medias_top(HASHTAG, amount=TOP_MEDIA_AMOUNT):

    media_id = media.pk
    caption = safe_caption(media)

    # fetch comments & insights
    comments = cl.media_comments(media_id, amount=MAX_COMMENTS)
    insights = cl.insights_media(media_id)
    metrics = parse_insights(insights)

    # pick most-liked comments
    sorted_comments = sorted(comments, key=lambda c: (c.like_count or 0), reverse=True)[
        :TOP_COMMENTS
    ]

    for c in sorted_comments:
        post_data.append(
            {
                "Media ID": media_id,
                "Caption": caption,
                # ────────── post-level metrics ──────────
                "Post Likes": metrics.get("like_count", 0),  # NEW name
                "Views": metrics.get("impression_count", 0),
                "Saves": metrics.get(
                    "save_count", metrics.get("saved_count", 0)  # ### NEW
                ),
                "Shares": metrics.get("share_count", 0),
                "Num Comments": metrics.get("comment_count", 0),
                # ────────── comment-level metrics ───────
                "Comment Text": c.text,
                "Comment Likes": c.like_count,
                "Comment User": c.user.username,
            }
        )

    random_delay()

# ───────────────────────────────
# 5.  SAVE  +  SUMMARY
# ───────────────────────────────
df = pd.DataFrame(post_data)
df.to_csv("instagram_hashtag_metrics.csv", index=False)
print("Saved → instagram_hashtag_metrics.csv")

# Quick console summary (optional)
totals = (
    df[["Post Likes", "Views", "Saves", "Shares", "Num Comments"]]
    .drop_duplicates()  # one row per media, not per top comment
    .sum()
)
print("\nAggregate across the top", TOP_MEDIA_AMOUNT, "posts:")
for k, v in totals.items():
    print(f"  {k:13}: {v:,}")
