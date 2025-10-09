# analysis.py
import pandas as pd
import numpy as np
from scipy.stats import zscore
from prophet import Prophet
import re

def unify_and_concat(news_df: pd.DataFrame, twitter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns and concat. Output columns: timestamp, text, sentiment_compound, source
    """
    news = news_df.copy() if news_df is not None else pd.DataFrame()
    twitter = twitter_df.copy() if twitter_df is not None else pd.DataFrame()

    # detect timestamp-like column for each
    def pick_time_col(df):
        if df is None or df.shape[0] == 0:
            return None
        for c in df.columns:
            if c.lower() in ("timestamp","datetime","date","published_date","created_at","publishedat","createdat"):
                return c
        return None

    n_ts = pick_time_col(news)
    t_ts = pick_time_col(twitter)

    if n_ts:
        news = news.rename(columns={n_ts: "timestamp"})
    if "text" not in news.columns:
        if "description" in news.columns:
            news["text"] = news["description"].fillna("")
        elif "title" in news.columns:
            news["text"] = news["title"].fillna("")
        else:
            news["text"] = ""

    if t_ts:
        twitter = twitter.rename(columns={t_ts: "timestamp"})
    if "text" not in twitter.columns:
        if "text" not in twitter.columns and "content" in twitter.columns:
            twitter["text"] = twitter["content"].fillna("")
        elif "text" not in twitter.columns and "title" in twitter.columns:
            twitter["text"] = twitter["title"].fillna("")
        else:
            twitter["text"] = ""

    if "sentiment_compound" not in news.columns:
        news["sentiment_compound"] = np.nan
    if "sentiment_compound" not in twitter.columns:
        twitter["sentiment_compound"] = np.nan

    news["source"] = "news"
    twitter["source"] = "twitter"

    cols = ["timestamp", "text", "sentiment_compound", "source"]
    combined = pd.concat([news[cols], twitter[cols]], ignore_index=True, sort=False)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    combined["sentiment_compound"] = combined["sentiment_compound"].fillna(0.0)
    return combined

def aggregate_timeseries(events_df: pd.DataFrame, freq: str = "H", roll_window: int = 3, keywords=None) -> pd.DataFrame:
    """
    Aggregate events into time buckets. Returns DataFrame with columns:
    timestamp, sentiment_mean, count, sent_smoothed, count_z, (kw__<kw>, kw__<kw>_z...)
    """
    if events_df is None or events_df.shape[0] == 0:
        return pd.DataFrame(columns=["timestamp","sentiment_mean","count","sent_smoothed"])

    events = events_df.copy()
    events["period"] = events["timestamp"].dt.floor(freq)
    agg = events.groupby("period").agg(sentiment_mean=("sentiment_compound","mean"),
                                       count=("sentiment_compound","size")).reset_index().rename(columns={"period":"timestamp"})
    full_index = pd.date_range(start=agg["timestamp"].min(), end=agg["timestamp"].max(), freq=freq)
    agg = agg.set_index("timestamp").reindex(full_index).fillna({"sentiment_mean":0,"count":0}).reset_index().rename(columns={"index":"timestamp"})
    agg["sent_smoothed"] = agg["sentiment_mean"].rolling(roll_window, min_periods=1).mean()

    # keyword monitoring
    if keywords:
        texts_per_period = events.groupby(pd.Grouper(key="timestamp", freq=freq))["text"].apply(lambda x: " ".join(x)).reindex(full_index).fillna("")
        for kw in keywords:
            pat = rf"(?i)\b{re.escape(kw)}\b"
            agg[f"kw__{kw}"] = texts_per_period.str.count(pat).values
            col = f"kw__{kw}"
            agg[f"{col}_z"] = zscore(agg[col]) if agg[col].std() > 0 else 0.0

    agg["count_z"] = zscore(agg["count"]) if agg["count"].std() > 0 else 0.0
    return agg

def forecast_sentiment(agg_df: pd.DataFrame, periods: int = 24, freq: str = "H"):
    """
    Forecast using Prophet. Returns forecast DataFrame or None if not enough data.
    """
    if agg_df is None or agg_df.shape[0] < 5:
        return None
    df = agg_df[["timestamp", "sent_smoothed"]].rename(columns={"timestamp":"ds","sent_smoothed":"y"}).copy()
    df = df.dropna(subset=["ds","y"])
    if df.shape[0] < 5:
        return None
    m = Prophet(seasonality_mode="additive", weekly_seasonality=True, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast
