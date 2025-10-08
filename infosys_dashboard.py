import os
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from gnews import GNews
import tweepy
from transformers import pipeline
from prophet import Prophet
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

nltk.download("vader_lexicon", quiet=True)



# DATA FETCHING (GOOGLE NEWS + TWITTER)


def fetch_google_news(query, limit=20):
    google_news = GNews(language="en", max_results=limit)
    news_items = google_news.get_news(query)
    news_list = []
    for item in news_items:
        news_list.append([
            item.get("title"),
            item.get("published date"),
            item.get("description"),
            item.get("url"),
        ])
    df_news = pd.DataFrame(news_list, columns=["title", "published_date", "description", "url"])
    return df_news

# Twitter Bearer Token - Replace with your own!!!
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIUd4QEAAAAABmjqcTUcSSIoQ7wqer1Qq5XBR8g%3DtWXegMV4TtDuBcGtmIzMkUVDWi1Bz7ukz9W81LNBedWV0zvnof"

def fetch_twitter_data_tweepy(query, limit=10, bearer_token=BEARER_TOKEN):
    client = tweepy.Client(bearer_token)
    response = client.search_recent_tweets(
        query=query,
        tweet_fields=["created_at", "author_id"],
        expansions="author_id",
        user_fields="username",
        max_results=min(limit, 10)
    )
    tweets_list = []
    users = {user["id"]: user["username"] for user in response.includes.get("users", [])}
    for tweet in response.data:
        tweets_list.append([
            tweet.created_at,
            tweet.id,
            tweet.text,
            users.get(tweet.author_id, "unknown"),
            f"https://twitter.com/{users.get(tweet.author_id, 'unknown')}/status/{tweet.id}"
        ])
    df_twitter = pd.DataFrame(tweets_list, columns=["date", "id", "content", "username", "url"])
    return df_twitter

def ensure_local_csv(fname, frame):
    if not os.path.exists(fname):
        frame.to_csv(fname, index=False)
    else:
        try:
            tmp = pd.read_csv(fname)
            print(f"Loaded existing: {fname}")
            return tmp
        except Exception:
            frame.to_csv(fname, index=False)
    return frame


# SENTIMENT ANALYSIS


def sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment_score(text, sentiment_analyzer):
    if not text or text.strip() == "":
        return ("NEUTRAL", 0.0, 0.0)
    try:
        result = sentiment_analyzer(text[:512])[0]  # truncate long text
        label = result["label"]
        score = result["score"]
        numeric = score if label == "POSITIVE" else -score if label == "NEGATIVE" else 0.0
        return (label, score, round(numeric, 3))
    except:
        return ("NEUTRAL", 0.0, 0.0)

def assign_sentiment(df, text_column, sentiment_analyzer):
    labels, scores, numeric_values = [], [], []
    for text in df[text_column].fillna("").tolist():
        label, score, numeric = get_sentiment_score(text, sentiment_analyzer)
        labels.append(label)
        scores.append(score)
        numeric_values.append(numeric)
    df["sentiment_label"] = labels
    df["sentiment_confidence"] = scores
    df["sentiment_value"] = numeric_values
    return df


# AGGREGATION + FORECAST + ALERTS


def clean_rename(df):
    # Timestamp, Text, Sentiment
    colmap = {}
    for col in df.columns:
        cl = col.lower()
        if "date" in cl or "time" in cl:
            colmap[col] = "timestamp"
        elif "title" in cl or "content" in cl or "text" in cl or "desc" in cl:
            colmap[col] = "text"
        elif "sent" in cl:
            colmap[col] = "sentiment_value"
    df = df.rename(columns=colmap)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def vadersentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_value'] = df["text"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
    return df

def aggregate_events(dfs, agg_freq='H', roll_window=3, monitor_kw=["AI"]):
    events = pd.concat([
        dfs['news'][["timestamp", "text", "sentiment_value"]].assign(source="news"),
        dfs['twitter'][["timestamp", "text", "sentiment_value"]].assign(source="twitter")
    ])
    events = events.sort_values("timestamp")
    events["period"] = events["timestamp"].dt.floor(agg_freq)
    agg = events.groupby("period").agg(
        sentiment_value=("sentiment_value", "mean"),
        count=("sentiment_value", "size")
    ).reset_index().rename(columns={"period": "timestamp"})
    full_index = pd.date_range(start=agg["timestamp"].min(), end=agg["timestamp"].max(), freq=agg_freq)
    agg = agg.set_index("timestamp").reindex(full_index).fillna({"sentiment_value": 0, "count": 0}).reset_index()
    agg = agg.rename(columns={"index": "timestamp"})
    agg["sent_smoothed"] = agg["sentiment_value"].rolling(roll_window, min_periods=1).mean()
    # Keyword monitoring
    texts_per_period = events.groupby(pd.Grouper(key="timestamp", freq=agg_freq))["text"].apply(lambda x: " ".join(x)).reindex(full_index).fillna("")
    for kw in monitor_kw:
        agg[f"kw__{kw}"] = texts_per_period.str.count(rf"\b{kw}\b", flags=re.IGNORECASE).values
    from scipy.stats import zscore
    agg["count_z"] = zscore(agg["count"]) if agg["count"].std() > 0 else 0
    for kw in monitor_kw:
        col = f"kw__{kw}"
        agg[f"{col}_z"] = zscore(agg[col]) if agg[col].std() > 0 else 0
    return agg

def forecast_sentiment(agg, forecast_periods=24, agg_freq='H'):
    df = agg.rename(columns={"timestamp": "ds", "sentiment_value": "y"})
    df["y"] = df["y"] * 100  # scale
    if len(df) >= 5:
        m = Prophet(seasonality_mode="additive", weekly_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=forecast_periods, freq=agg_freq)
        forecast = m.predict(future)
        return forecast
    else:
        return None


# STRATEGIC INTELLIGENCE DASHBOARD


def plot_dashboard(agg, forecast=None, monitor_kw=["AI"]):
    st.title("Strategic Intelligence Dashboard")

    # Main sentiment timeline
    st.header("Sentiment Timeline")
    fig, ax = plt.subplots(figsize=(13,5))
    sns.lineplot(x='timestamp', y='sent_smoothed', data=agg, marker='o', ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Smoothed Sentiment")
    st.pyplot(fig)

    # Forecast
    if forecast is not None:
        st.header("Sentiment Forecast")
        fig2, ax2 = plt.subplots(figsize=(13,5))
        sns.lineplot(x='ds', y='yhat', data=forecast, marker='o', label='Forecast', ax=ax2)
        plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, color='orange', label='Forecast CI')
        ax2.set_title("Sentiment Forecast")
        ax2.set_xlabel("Time")
        ax2.legend()
        st.pyplot(fig2)

    # Volume
    st.header("Message Volume")
    fig3, ax3 = plt.subplots(figsize=(13,3))
    sns.barplot(x='timestamp', y='count', data=agg, color='skyblue', ax=ax3)
    plt.xticks(rotation=45)
    ax3.set_ylabel("Volume")
    st.pyplot(fig3)

    # Keyword Z-Score
    st.header("Keyword Spike Detection")
    for kw in monitor_kw:
        col = f"kw__{kw}_z"
        if col in agg.columns:
            figk, axk = plt.subplots(figsize=(13,2))
            sns.lineplot(x='timestamp', y=col, data=agg, marker='o', ax=axk)
            axk.axhline(2.0, ls="--", c="r", label="Spike Threshold")
            axk.legend()
            st.pyplot(figk)
    # Alert Table
    st.header("Recent Alerts")
    alert_events = agg[
        (agg['sent_smoothed'] <= -0.3) | (agg['sent_smoothed'] >= 0.3) | (agg['count_z'] >= 2.0)
    ]
    if not alert_events.empty:
        st.dataframe(alert_events.head(10))
    else:
        st.markdown("No alert-worthy events detected recently.")


# Main

def main():
    st.sidebar.title("Control Panel")
    st.sidebar.markdown("**Choose a topic & timeframe, then run analysis**")
    query = st.sidebar.text_input("Topic/Keyword", "Artificial Intelligence")
    fetch = st.sidebar.button("Fetch & Analyze Data")

    if fetch:
        news_df = fetch_google_news(query, limit=20)
        news_df = ensure_local_csv("news_with_sentiment.csv", news_df)
        twitter_df = fetch_twitter_data_tweepy(query, limit=20)
        twitter_df = ensure_local_csv("twitter_with_sentiment.csv", twitter_df)

        # Sentiment
        sentiment_analyzer = sentiment_model()
        news_df = assign_sentiment(news_df, "title", sentiment_analyzer)
        twitter_df = assign_sentiment(twitter_df, "content", sentiment_analyzer)

        # Local save
        news_df.to_csv("news_with_sentiment.csv", index=False)
        twitter_df.to_csv("twitter_with_sentiment.csv", index=False)

        # Timestamp fix
        news_df = clean_rename(news_df)
        twitter_df = clean_rename(twitter_df)
        news_df = vadersentiment(news_df)
        twitter_df = vadersentiment(twitter_df)

        monitor_kw = [query.split()[0]]  # Basic choice
        agg = aggregate_events({'news': news_df, 'twitter': twitter_df}, monitor_kw=monitor_kw)

        # Save aggregation
        agg.to_csv("aggregated_series.csv", index=False)

        forecast = forecast_sentiment(agg)
        if forecast is not None:
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv("sentiment_forecast.csv", index=False)

        plot_dashboard(agg, forecast, monitor_kw=monitor_kw)

if __name__ == "__main__":
    main()
