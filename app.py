# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from data_fetch import fetch_google_news, fetch_twitter_recent
from sentiment_utils import add_vader_sentiment
from analysis import unify_and_concat, aggregate_timeseries, forecast_sentiment
from utils import send_slack
import plotly.express as px

load_dotenv()

st.set_page_config(page_title="Strategic Intelligence Dashboard", layout="wide")

st.title("Strategic Intelligence Dashboard — Celebrare Internship")
st.markdown("📊 Fetch Google News + Twitter → Analyze Sentiment → Monitor Keywords → Forecast Trends → Export Results")

# Sidebar settings
st.sidebar.header("Configuration")
query = st.sidebar.text_input("Query / Topic", value="Artificial Intelligence")
news_limit = st.sidebar.number_input("News items to fetch", min_value=5, max_value=200, value=50)
tweet_limit = st.sidebar.number_input("Tweets to fetch", min_value=5, max_value=100, value=50)
agg_freq = st.sidebar.selectbox("Aggregation frequency", options=["H","D"], index=0, format_func=lambda x: "Hourly" if x=="H" else "Daily")
roll_window = st.sidebar.number_input("Smoothing window (periods)", min_value=1, max_value=24, value=3)
keywords_raw = st.sidebar.text_input("Keywords (comma-separated)", value=os.getenv("DEFAULT_KEYWORDS","AI,ML"))
keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
run_fetch = st.sidebar.button("Fetch & Refresh Now")

# Load or fetch data
@st.cache_data(ttl=300)
def fetch_and_process(topic, n_news, n_tweets):
    news_df = fetch_google_news(topic, limit=n_news)
    tweet_df = fetch_twitter_recent(topic, limit=n_tweets)

    # Ensure text columns exist
    if "description" in news_df.columns and "title" in news_df.columns:
        news_df["text"] = news_df["description"].fillna("") + " " + news_df["title"].fillna("")
    elif "title" in news_df.columns:
        news_df["text"] = news_df["title"].fillna("")
    if "text" not in tweet_df.columns and "content" in tweet_df.columns:
        tweet_df["text"] = tweet_df["content"]

    # Add sentiment
    news_df = add_vader_sentiment(news_df, "text")
    tweet_df = add_vader_sentiment(tweet_df, "text")
    return news_df, tweet_df

if run_fetch:
    st.info("🔄 Fetching fresh data...")
news_df, tweet_df = fetch_and_process(query, news_limit, tweet_limit)

# --- Raw Data ---
st.subheader("📰 Raw Data Preview")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**News (sample)**")
    st.dataframe(news_df.head(10))
    st.download_button("⬇️ Download News Data (CSV)", news_df.to_csv(index=False), "news_data.csv", "text/csv")
with col2:
    st.markdown("**Tweets (sample)**")
    st.dataframe(tweet_df.head(10))
    st.download_button("⬇️ Download Tweet Data (CSV)", tweet_df.to_csv(index=False), "tweet_data.csv", "text/csv")

# --- Combine & Aggregate ---
events = unify_and_concat(news_df, tweet_df)
st.subheader("🧩 Combined Events")
st.dataframe(events.head(20))
st.download_button("⬇️ Download Combined Data (CSV)", events.to_csv(index=False), "combined_data.csv", "text/csv")

agg = aggregate_timeseries(events, freq=agg_freq, roll_window=roll_window, keywords=keywords)
st.subheader("📈 Aggregated Timeseries")
st.dataframe(agg.tail(50))
st.download_button("⬇️ Download Aggregated Data (CSV)", agg.to_csv(index=False), "aggregated_data.csv", "text/csv")

# --- Alerts ---
st.subheader("🚨 Alerts")
NEG_SENT_THRESHOLD = st.sidebar.slider("Negative sentiment threshold", -1.0, 0.0, -0.3, 0.05)
POS_SENT_THRESHOLD = st.sidebar.slider("Positive sentiment threshold", 0.0, 1.0, 0.3, 0.05)
VOLUME_Z_THRESHOLD = st.sidebar.slider("Volume spike z-threshold", 0.5, 5.0, 2.0, 0.1)

alerts = []
latest = agg.iloc[-1]
if latest["sent_smoothed"] <= NEG_SENT_THRESHOLD:
    alerts.append(("Negative sentiment", latest["sent_smoothed"]))
if latest["sent_smoothed"] >= POS_SENT_THRESHOLD:
    alerts.append(("Positive sentiment", latest["sent_smoothed"]))
if latest["count_z"] >= VOLUME_Z_THRESHOLD:
    alerts.append(("Volume spike", latest["count_z"]))

# Keyword spike alerts
for kw in keywords:
    col_z = f"kw__{kw}_z"
    if col_z in agg.columns and agg[col_z].iloc[-1] >= VOLUME_Z_THRESHOLD:
        alerts.append((f"Keyword spike: {kw}", agg[col_z].iloc[-1]))

if alerts:
    for a in alerts:
        st.error(f"⚠️ {a[0]} — value {a[1]:.2f}")
    if st.sidebar.button("📤 Send alerts to Slack"):
        webhook = os.getenv("SLACK_WEBHOOK_URL","")
        for a in alerts:
            send_slack(webhook, f"ALERT: {a[0]} value {a[1]:.2f}")
        st.success("✅ Alerts sent to Slack (if webhook provided).")
else:
    st.success("✅ No alerts currently.")

# --- Charts ---
st.subheader("📊 Sentiment Trend")
fig = px.line(agg, x="timestamp", y="sent_smoothed", title="Smoothed Sentiment Over Time", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔑 Keyword Trends")
for kw in keywords:
    col = f"kw__{kw}"
    if col in agg.columns:
        fig2 = px.line(agg, x="timestamp", y=col, title=f"Keyword frequency: {kw}")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("📦 Message Volume")
fig3 = px.bar(agg, x="timestamp", y="count", title="Volume Over Time")
st.plotly_chart(fig3, use_container_width=True)

# --- Forecast ---
st.subheader("🔮 Forecast (Prophet)")
forecast_periods = st.sidebar.number_input("Forecast periods", min_value=1, max_value=168, value=24)
if st.sidebar.button("Run Forecast"):
    with st.spinner("Running Prophet forecast..."):
        fcst = forecast_sentiment(agg, periods=int(forecast_periods), freq=agg_freq)
        if fcst is None:
            st.warning("Not enough history for forecasting.")
        else:
            fig_fc = px.line(fcst, x="ds", y="yhat", title="Forecast (Predicted Sentiment)")
            st.plotly_chart(fig_fc, use_container_width=True)
            st.dataframe(fcst[["ds","yhat","yhat_lower","yhat_upper"]].tail(20))
            st.download_button("⬇️ Download Forecast Data (CSV)", fcst.to_csv(index=False), "forecast_data.csv", "text/csv")
