# data_fetch.py
import os
import re
import pandas as pd
from gnews import GNews
import tweepy
from dotenv import load_dotenv
from dateutil import parser

load_dotenv()
TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN", "")

def fetch_google_news(query: str, limit: int = 50) -> pd.DataFrame:
    """
    Fetch news items using GNews. Returns DataFrame with columns:
    ['source', 'title', 'published_date', 'description', 'url']
    """
    google = GNews(language="en", max_results=min(limit, 100))
    try:
        items = google.get_news(query)
    except Exception as e:
        print("GNews error:", e)
        items = []

    rows = []
    for it in items:
        title = it.get("title") or ""
        # Different versions may have different published fields
        published = it.get("published date") or it.get("published") or it.get("publishedAt") or None
        try:
            published_dt = parser.parse(published) if published else pd.NaT
        except Exception:
            published_dt = pd.NaT
        rows.append({
            "source": "news",
            "title": title,
            "published_date": published_dt,
            "description": it.get("description") or "",
            "url": it.get("url") or ""
        })
    df = pd.DataFrame(rows)
    return df

def _init_twitter_client():
    if not TWITTER_BEARER:
        raise EnvironmentError("TWITTER_BEARER_TOKEN not found in environment (.env).")
    return tweepy.Client(bearer_token=TWITTER_BEARER, wait_on_rate_limit=True)

def fetch_twitter_recent(query: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetch recent tweets using Tweepy v4 Paginator.
    Returns DataFrame with columns:
    ['source','created_at','id','text','username','url']
    """
    client = _init_twitter_client()
    tweets = []
    collected = 0
    # Use Paginator to safely fetch multiple pages (max 100 per request)
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["id","text","created_at","author_id","lang"],
        expansions=["author_id"],
        user_fields=["username"],
        max_results=100
    )
    users_map = {}
    for resp in paginator:
        if resp is None:
            continue
        # map user ids -> username (resp.includes may be None or contain dict/list)
        includes = getattr(resp, "includes", None) or resp.get("includes", {}) if isinstance(resp, dict) else {}
        # includes users may be list of tweepy.User or dicts
        users = []
        if isinstance(includes, dict) and "users" in includes:
            users = includes["users"]
        elif hasattr(resp, "includes") and getattr(resp, "includes") and "users" in resp.includes:
            users = resp.includes["users"]
        for u in users:
            uid = getattr(u, "id", None) or (u.get("id") if isinstance(u, dict) else None)
            uname = getattr(u, "username", None) or (u.get("username") if isinstance(u, dict) else None)
            if uid:
                users_map[str(uid)] = uname or "unknown"

        data_items = getattr(resp, "data", None) or resp.get("data", []) if isinstance(resp, dict) else getattr(resp, "data", None)
        if not data_items:
            continue
        for t in data_items:
            if collected >= limit:
                break
            # t may be tweepy.Tweet or dict
            created = getattr(t, "created_at", None) or (t.get("created_at") if isinstance(t, dict) else None)
            text = getattr(t, "text", None) or (t.get("text") if isinstance(t, dict) else "")
            tid = getattr(t, "id", None) or (t.get("id") if isinstance(t, dict) else None)
            author_id = getattr(t, "author_id", None) or (t.get("author_id") if isinstance(t, dict) else None)
            username = users_map.get(str(author_id), "unknown")
            tweets.append({
                "source": "twitter",
                "created_at": created,
                "id": tid,
                "text": text,
                "username": username,
                "url": f"https://twitter.com/{username}/status/{tid}"
            })
            collected += 1
        if collected >= limit:
            break

    df = pd.DataFrame(tweets)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df
